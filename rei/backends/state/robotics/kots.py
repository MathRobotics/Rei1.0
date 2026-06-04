from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np

from ....core.state_cache import StateKey
from ....core.state_schema import (
    DYNAMICS_FIELDS,
    DTYPE_COORD,
    DTYPE_DYNAMICS,
    DTYPE_KINEMATICS,
    canonical_field_name,
    split_jac_field,
    torque_derivative_field,
    torque_derivative_order,
)
from ....core.trajectory import TrajectoryMap
from ..dispatch.template import BackendDispatchStateBuilder
from .motion import (
    dof_sorted_robot_joints,
    expand_coordinate_motion_by_robot_layout,
    infer_robot_model_dof,
    infer_robot_model_order,
    interleaved_motion_jacobian_used_order,
)
from .trajectory import (
    chain_param_jacobian,
    compose_interleaved_motion_and_jac,
    TrajectoryStateBuilderMixin,
    unique_jacobian_chain_candidates,
    validate_trajectory_derivative_maps,
)
from . import kots_api as kapi
from .kots_api import StateType

Array = np.ndarray
STATE_JACOBIAN_VAR = "state"
_KOTS_JACOBIAN_STRATEGIES = ("dense", "mul")


def _normalize_kots_jacobian_strategy(strategy: str | None, *, prefer_matvec_jacobian: bool) -> str:
    if prefer_matvec_jacobian:
        warnings.warn(
            "KotsTrajectoryStateBuilder: prefer_matvec_jacobian is deprecated; "
            "use jacobian_strategy='mul' or 'dense' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if strategy is None:
        return "mul"
    name = str(strategy).strip().lower()
    if name == "":
        raise ValueError("KotsTrajectoryStateBuilder: jacobian_strategy must be non-empty.")
    if name == "matvec":
        name = "mul"
    if name not in _KOTS_JACOBIAN_STRATEGIES:
        allowed = ", ".join(repr(v) for v in _KOTS_JACOBIAN_STRATEGIES)
        raise ValueError(f"KotsTrajectoryStateBuilder: jacobian_strategy must be one of {allowed}, got {strategy!r}.")
    return name


@dataclass(frozen=True)
class KotsFieldFamily:
    field: str


@dataclass(frozen=True)
class _TotalJointDynamicsStateRef:
    field: str
    refs: tuple[Any, ...]


# kots.py 内で「どの field ファミリを提供するか」を宣言する登録リスト。
KOTS_DEFAULT_FIELD_FAMILIES: tuple[KotsFieldFamily, ...] = (
    KotsFieldFamily(field="pos"),
    KotsFieldFamily(field="rot"),
    KotsFieldFamily(field="frame"),
)


class KotsStateBuilder(BackendDispatchStateBuilder):
    """RoboKots/Kots -> `build_state()` bridge with StateKey-based automatic dispatch."""

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        q_var: str = "q",
        fields: Sequence[str] | None = None,
        dynamics_fields: Sequence[str] | None = DYNAMICS_FIELDS,
        dynamics_owner_type: str = "total_joint",
        prefer_matvec_jacobian: bool = False,
    ) -> None:
        super().__init__(model, data, q_var=q_var)
        self.dtype = DTYPE_KINEMATICS
        self.owner_type = "link"
        self.dynamics_owner_type = str(dynamics_owner_type)
        if self.dynamics_owner_type == "":
            raise ValueError("KotsStateBuilder: dynamics_owner_type must be non-empty.")
        self._needs_dynamics_update = False
        self._model_dof_cache: int | None = None
        self._model_order_cache: int | None = None
        self.prefer_matvec_jacobian = bool(prefer_matvec_jacobian)
        self._jacobian_ops = kapi.RoboKotsJacobianOperator(self.model)

        family_map = {spec.field: spec for spec in KOTS_DEFAULT_FIELD_FAMILIES}
        selected_fields = [spec.field for spec in KOTS_DEFAULT_FIELD_FAMILIES] if fields is None else [str(f) for f in fields]
        if len(selected_fields) == 0:
            raise ValueError("KotsStateBuilder: fields must be non-empty.")

        self.field_to_jac: dict[str, str] = {}
        for field in selected_fields:
            spec = family_map.get(field, None)
            if spec is None:
                supported = ", ".join(sorted(family_map.keys()))
                raise ValueError(
                    f"KotsStateBuilder: unsupported field {field!r}. "
                    f"Supported fields: {supported}."
                )
            _value_name, jac_name = self.register_value_and_jac(
                dtype=self.dtype,
                owner_type=self.owner_type,
                field=spec.field,
                value_handler=self._handle_value,
                jac_handler=self._handle_jac,
                jacobian_wrt=STATE_JACOBIAN_VAR,
            )
            self.field_to_jac[spec.field] = jac_name

        if dynamics_fields is not None:
            dyn_fields_raw = [str(f) for f in dynamics_fields]
            dyn_fields = []
            for field in dyn_fields_raw:
                if field == "":
                    raise ValueError("KotsStateBuilder: dynamics field names must be non-empty.")
                dyn_fields.append(canonical_field_name(field))
            dyn_fields = list(dict.fromkeys(dyn_fields))
            if len(dyn_fields) == 0:
                raise ValueError("KotsStateBuilder: dynamics_fields must be non-empty when provided.")
            self._needs_dynamics_update = True
            for field in dyn_fields:
                self.register_value_and_jac(
                    dtype=DTYPE_DYNAMICS,
                    owner_type=self.dynamics_owner_type,
                    field=field,
                    value_handler=self._handle_value,
                    jac_handler=self._handle_jac,
                    jacobian_wrt=STATE_JACOBIAN_VAR,
                )

    def _update_dynamics_if_available(self) -> bool:
        if not self._needs_dynamics_update:
            return False
        for name in ("dynamics", "compute_dynamics", "update_dynamics"):
            fn = getattr(self.model, name, None)
            if not callable(fn):
                continue
            try:
                fn()
            except TypeError:
                continue
            return True
        return False

    def _update_kinematics(self, q: Array) -> None:
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        dof = self._model_dof()
        order = self._model_order()

        if q_vec.size == dof * order:
            motion = q_vec
        elif q_vec.size == dof:
            motion = self._expand_coordinate_motion(q_vec, dof=dof, order=order)
        else:
            raise ValueError(
                "KotsStateBuilder: unexpected q size. "
                f"Expected dof ({dof}) or dof*order ({dof * order}), got {q_vec.size}."
            )

        self.model.import_motions(motion)
        if self._needs_dynamics_update:
            if self._update_dynamics_if_available():
                return

        self.model.kinematics()
        if self._needs_dynamics_update:
            self._update_dynamics_if_available()

    def _model_dof(self) -> int:
        if self._model_dof_cache is not None:
            return int(self._model_dof_cache)
        dof = infer_robot_model_dof(self.model)
        self._model_dof_cache = int(dof)
        return int(dof)

    def _model_order(self) -> int:
        if self._model_order_cache is not None:
            return int(self._model_order_cache)
        order = infer_robot_model_order(self.model)
        self._model_order_cache = int(order)
        return int(order)

    def _expand_coordinate_motion(self, q: Array, *, dof: int, order: int) -> Array:
        return expand_coordinate_motion_by_robot_layout(
            self.model,
            q,
            dof=dof,
            order=order,
            error_prefix="KotsStateBuilder",
        )

    def _resolve_state_ref(self, key: StateKey) -> Any:
        owner = getattr(key, "owner", None)
        owner_type = getattr(owner, "owner_type", None)
        owner_name = getattr(owner, "owner_name", None)
        if not isinstance(owner_name, str) or owner_name == "":
            raise ValueError(f"Kots backend expects non-empty owner_name in key, got: {key!r}")

        state_field = self._state_field_name(key.field)
        if owner_type == "total_joint" and getattr(key, "dtype", None) == DTYPE_COORD and key.field == "q":
            # Joint-q terms are computed directly from optimization variables; no backend state query is required.
            return ("total_joint", owner_name, "q")

        if owner_type == self.dynamics_owner_type and getattr(key, "dtype", None) == DTYPE_DYNAMICS:
            # RoboKots does not robustly support world dynamics queries for owner_type="total_joint".
            # Expand to per-joint queries and stack them in dof order.
            joint_refs = self._resolve_total_joint_dynamics_refs(state_field=state_field, key=key)
            if joint_refs is not None:
                return _TotalJointDynamicsStateRef(field=state_field, refs=joint_refs)

        route = self._route_for_key(key)
        if route is None or route not in self._dispatch:
            raise ValueError(f"Kots backend has no handler route for key: {key!r}")

        frame_name = getattr(key, "frame", None) or "world"
        return self._make_state_type(
            owner_type=str(owner_type),
            owner_name=owner_name,
            state_field=state_field,
            frame_name=str(frame_name),
        )

    def _resolve_total_joint_dynamics_refs(self, *, state_field: str, key: StateKey) -> tuple[Any, ...] | None:
        joints = self._dof_sorted_joints()
        if joints is None:
            return None

        if len(joints) == 0:
            return tuple()

        frame_name = self._total_joint_dynamics_frame_name(state_field=state_field, key=key)

        refs: list[Any] = []
        for joint in joints:
            joint_name = str(getattr(joint, "name", ""))
            if joint_name == "":
                raise ValueError("KotsStateBuilder: joint.name must be non-empty for dynamics expansion.")
            refs.append(
                self._make_state_type(
                    owner_type="joint",
                    owner_name=joint_name,
                    state_field=state_field,
                    frame_name=frame_name,
                )
            )
        return tuple(refs)

    @staticmethod
    def _state_field_name(field: Any) -> str:
        return canonical_field_name(str(field))

    @staticmethod
    def _fallback_backend_field_name(state_field: str) -> str:
        return kapi.fallback_backend_field_name(state_field)

    def _make_state_type(
        self,
        *,
        owner_type: str,
        owner_name: str,
        state_field: str,
        frame_name: str | None,
    ) -> Any:
        return kapi.make_state_type(
            owner_type=owner_type,
            owner_name=owner_name,
            state_field=state_field,
            frame_name=frame_name,
            state_type=StateType,
        )

    @staticmethod
    def _torque_derivative_order_from_state_data_type(data_type: str) -> int | None:
        return kapi.torque_derivative_order_from_state_data_type(data_type)

    def _dof_sorted_joints(self) -> list[Any] | None:
        return dof_sorted_robot_joints(self.model)

    def _total_joint_dynamics_frame_name(self, *, state_field: str, key: StateKey) -> str | None:
        frame_name = getattr(key, "frame", None) or "world"
        # torque family does not need world-frame conversion in RoboKots get_value path.
        # Keeping frame unset avoids a known owner_type='total_joint' world-path bug upstream.
        if "torque" in state_field:
            return None
        return str(frame_name)

    @staticmethod
    def _state_ref_data_type(state_ref: Any) -> str | None:
        return kapi.state_ref_data_type(state_ref)

    @staticmethod
    def _state_ref_owner_type(state_ref: Any) -> str | None:
        return kapi.state_ref_owner_type(state_ref)

    @staticmethod
    def _state_ref_owner_name(state_ref: Any) -> str | None:
        return kapi.state_ref_owner_name(state_ref)

    @staticmethod
    def _state_ref_frame_name(state_ref: Any) -> str | None:
        return kapi.state_ref_frame_name(state_ref)

    def _raise_missing_state_key(self, *, state_ref: Any, cause: KeyError) -> None:
        data_type = self._state_ref_data_type(state_ref)
        deriv_order = None if data_type is None else self._torque_derivative_order_from_state_data_type(data_type)
        if isinstance(deriv_order, int) and deriv_order > 0:
            field = torque_derivative_field(deriv_order)
            required_model_order = deriv_order + 3
            raise ValueError(
                "KotsStateBuilder: dynamics field "
                f"{field!r} requires RoboKots model order >= {required_model_order}. "
                f"Current model order is {self._model_order()}."
            ) from cause
        raise cause

    def _value_from_single_state_ref(self, state_ref: Any) -> Array:
        try:
            return kapi.state_info(self.model, state_ref)
        except KeyError as e:
            self._raise_missing_state_key(state_ref=state_ref, cause=e)

    def _value_from_state_ref(self, state_ref: Any) -> Array:
        total_joint_ref = self._as_total_joint_dynamics_state_ref(state_ref)
        if total_joint_ref is not None:
            return self._concat_total_joint_values(total_joint_ref.refs)

        return self._value_from_single_state_ref(state_ref)

    def _value_from_state_refs(self, refs: tuple[Any, ...]) -> Array:
        if len(refs) == 0:
            return np.zeros((0,), dtype=float)

        values = kapi.state_info_list(self.model, refs)
        if values is not None:
            return values

        return self._concat_total_joint_values(refs)

    def _jac_from_single_state_ref(self, state_ref: Any) -> Array:
        try:
            J = self._jacobian_ops.dense(state_ref)
        except KeyError as e:
            self._raise_missing_state_key(state_ref=state_ref, cause=e)
        return self._normalize_jacobian_shape(J, state_ref=state_ref)

    def _normalize_jacobian_shape(self, J_raw: Any, *, state_ref: Any) -> Array:
        J = np.asarray(J_raw, dtype=float)
        if J.ndim != 2:
            raise ValueError(f"Kots Jacobian must be 2D, got shape {J.shape}.")

        # Fast path: if one axis matches motion-space dimension, avoid an extra
        # state_info() call just for orientation inference.
        n_motion = int(self._model_dof() * self._model_order())
        if J.shape[1] == n_motion:
            return J
        if J.shape[0] == n_motion:
            return J.T

        total_joint_ref = self._as_total_joint_dynamics_state_ref(state_ref)
        if total_joint_ref is None:
            m = int(self._value_from_single_state_ref(state_ref).size)
        else:
            m = int(self._value_from_state_refs(total_joint_ref.refs).size)
        if J.shape[0] == m:
            return J
        if J.shape[1] == m:
            return J.T
        raise ValueError(f"Kots Jacobian must be ({m},n) or (n,{m}), got {J.shape}.")

    def _jac_from_state_refs(self, refs: tuple[Any, ...]) -> Array:
        if len(refs) == 0:
            return np.zeros((0, self._model_dof() * self._model_order()), dtype=float)

        J = self._jacobian_ops.dense_list(refs)
        if J is not None:
            return self._normalize_jacobian_shape(J, state_ref=_TotalJointDynamicsStateRef(field="", refs=refs))
        return self._stack_total_joint_jacobians_fallback(refs)

    def _matvec_from_single_state_ref(self, state_ref: Any, vec: Array) -> Array:
        return self._jacobian_ops.jvp(state_ref, vec)

    def _jac_from_matrix_mul_single_state_ref(self, state_ref: Any, cols: Array) -> Array:
        return self._jacobian_ops.jvp(state_ref, cols)

    def _jac_from_matvec_single_state_ref(self, state_ref: Any, cols: Array) -> Array:
        C = np.asarray(cols, dtype=float)
        if C.ndim != 2:
            raise ValueError(f"Kots matvec columns must be 2D, got shape {C.shape}.")
        try:
            m = int(self._value_from_single_state_ref(state_ref).size)
            return self._jacobian_ops.jvp(state_ref, C, value_size=m)
        except KeyError as e:
            self._raise_missing_state_key(state_ref=state_ref, cause=e)

    def _handle_value(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del q, key
        return self._value_from_state_ref(state_ref)

    def _handle_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del q
        total_joint_ref = self._as_total_joint_dynamics_state_ref(state_ref)
        if total_joint_ref is not None:
            return self._stack_total_joint_jacobians(total_joint_ref.refs)

        J = self._jac_from_single_state_ref(state_ref)
        return self._rotate_link_kinematics_jacobian_to_world(J=J, key=key, state_ref=state_ref)

    def _link_world_rotation(self, *, state_ref: Any) -> Array | None:
        owner_type = self._state_ref_owner_type(state_ref)
        if owner_type != self.owner_type:
            return None

        owner_name = self._state_ref_owner_name(state_ref)
        if not isinstance(owner_name, str) or owner_name == "":
            return None

        frame_name = self._state_ref_frame_name(state_ref) or "world"
        try:
            rot_ref = self._make_state_type(
                owner_type=owner_type,
                owner_name=owner_name,
                state_field="rot",
                frame_name=frame_name,
            )
            rot_flat = self._value_from_single_state_ref(rot_ref)
        except Exception:
            return None

        if int(rot_flat.size) != 9:
            return None
        return np.asarray(rot_flat, dtype=float).reshape(3, 3)

    def _rotate_link_kinematics_jacobian_to_world(
        self,
        *,
        J: Array,
        key: StateKey,
        state_ref: Any,
    ) -> Array:
        if getattr(key, "dtype", None) != DTYPE_KINEMATICS:
            return J
        owner = getattr(key, "owner", None)
        if getattr(owner, "owner_type", None) != self.owner_type:
            return J

        rot = self._link_world_rotation(state_ref=state_ref)
        if rot is None:
            return J

        rows = int(J.shape[0])
        if rows % 3 != 0:
            return J

        # RoboKots reports link Jacobians in the link-local frame.
        # Convert each 3D block into world-frame coordinates for backend parity.
        J_world = np.asarray(J, dtype=float).copy()
        for i in range(0, rows, 3):
            J_world[i : i + 3, :] = rot @ J_world[i : i + 3, :]
        return J_world

    def _rotate_link_kinematics_rhs_to_local(
        self,
        *,
        rhs: Array,
        key: StateKey,
        state_ref: Any,
    ) -> Array:
        if getattr(key, "dtype", None) != DTYPE_KINEMATICS:
            return rhs
        owner = getattr(key, "owner", None)
        if getattr(owner, "owner_type", None) != self.owner_type:
            return rhs

        rot = self._link_world_rotation(state_ref=state_ref)
        if rot is None:
            return rhs

        R = np.asarray(rhs, dtype=float)
        rows = int(R.shape[0])
        if rows % 3 != 0:
            return R

        R_local = R.copy()
        for i in range(0, rows, 3):
            R_local[i : i + 3, ...] = rot.T @ R_local[i : i + 3, ...]
        return R_local

    def _transpose_matvec_from_single_state_ref(self, state_ref: Any, rhs: Array) -> Array:
        return self._jacobian_ops.vjp(state_ref, rhs)

    def _transpose_matvec_from_state_ref(self, state_ref: Any, rhs: Array) -> Array:
        total_joint_ref = self._as_total_joint_dynamics_state_ref(state_ref)
        R = np.asarray(rhs, dtype=float)
        if R.ndim not in (1, 2):
            raise ValueError(f"Kots jacobian_transpose_mul rhs must be 1D or 2D, got shape {R.shape}.")

        if total_joint_ref is None:
            return self._transpose_matvec_from_single_state_ref(state_ref, R)

        try:
            return np.asarray(self._transpose_matvec_from_single_state_ref(list(total_joint_ref.refs), R), dtype=float)
        except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
            pass

        parts: list[Array] = []
        offset = 0
        for ref in total_joint_ref.refs:
            m = int(self._value_from_single_state_ref(ref).size)
            stop = offset + m
            if stop > int(R.shape[0]):
                raise ValueError(
                    "KotsStateBuilder: rhs is too short for total_joint transpose multiply. "
                    f"Need at least {stop} rows, got {R.shape[0]}."
                )
            parts.append(
                np.asarray(
                    self._transpose_matvec_from_single_state_ref(ref, R[offset:stop, ...]),
                    dtype=float,
                )
            )
            offset = stop

        if offset != int(R.shape[0]):
            raise ValueError(
                "KotsStateBuilder: rhs size mismatch for total_joint transpose multiply. "
                f"Consumed {offset} rows, got {R.shape[0]}."
            )
        if len(parts) == 0:
            return np.zeros((self._model_dof() * self._model_order(),), dtype=float)
        return np.sum(np.stack(parts, axis=0), axis=0)

    def jacobian_transpose_mul(
        self,
        q: Array,
        key: StateKey,
        rhs: Array,
        *,
        update_kinematics: bool = True,
    ) -> Array:
        """Compute `J(key).T @ rhs` with RoboKots' transpose multiply when available."""

        if not self._accept_required_key(key):
            raise ValueError(f"KotsStateBuilder: invalid state key for jacobian_transpose_mul: {key!r}")
        route = self._route_for_key(key)
        entry = None if route is None else self._dispatch.get(route, None)
        if entry is None:
            raise ValueError(f"KotsStateBuilder: no value handler route for key: {key!r}")

        q_vec = self._extract_q(np.asarray(q, dtype=float).reshape(-1), pack=None)
        if update_kinematics:
            self._update_kinematics(q_vec)
        state_ref = self._state_ref(key, state_ref_field=entry.state_ref_field)
        rhs_local = self._rotate_link_kinematics_rhs_to_local(
            rhs=np.asarray(rhs, dtype=float),
            key=key,
            state_ref=state_ref,
        )
        return self._transpose_matvec_from_state_ref(state_ref, rhs_local)

    @staticmethod
    def _as_total_joint_dynamics_state_ref(state_ref: Any) -> _TotalJointDynamicsStateRef | None:
        if isinstance(state_ref, _TotalJointDynamicsStateRef):
            return state_ref
        if (
            isinstance(state_ref, tuple)
            and len(state_ref) == 3
            and state_ref[0] == "total_joint_dynamics"
        ):
            return _TotalJointDynamicsStateRef(field=str(state_ref[1]), refs=tuple(state_ref[2]))
        return None

    def _concat_total_joint_values(self, refs: tuple[Any, ...]) -> Array:
        parts = [self._value_from_single_state_ref(ref) for ref in refs]
        if len(parts) == 0:
            return np.zeros((0,), dtype=float)
        return np.concatenate(parts, axis=0)

    def _stack_total_joint_jacobians(self, refs: tuple[Any, ...]) -> Array:
        return self._jac_from_state_refs(refs)

    def _stack_total_joint_jacobians_fallback(self, refs: tuple[Any, ...]) -> Array:
        blocks = [self._jac_from_single_state_ref(ref) for ref in refs]
        if len(blocks) == 0:
            return np.zeros((0, self._model_dof() * self._model_order()), dtype=float)
        ncols = int(blocks[0].shape[1])
        for block in blocks[1:]:
            if int(block.shape[1]) != ncols:
                raise ValueError(
                    "KotsStateBuilder: inconsistent Jacobian column size while stacking total_joint dynamics. "
                    f"Expected {ncols}, got {block.shape[1]}."
                )
        return np.vstack(blocks)


class KotsTrajectoryStateBuilder(TrajectoryStateBuilderMixin, KotsStateBuilder):
    """RoboKots trajectory builder with trajectory parameterization.

    Decision variable is `p` (configurable by `p_var`), and generalized coordinates are:

      q(k) = trajectory_map.q_at(p, k)
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        trajectory_map: TrajectoryMap,
        trajectory_derivative_maps: Mapping[int, TrajectoryMap] | None = None,
        p_var: str = "p",
        fields: Sequence[str] | None = None,
        dynamics_fields: Sequence[str] | None = DYNAMICS_FIELDS,
        dynamics_owner_type: str = "total_joint",
        prefer_matvec_jacobian: bool = False,
        jacobian_strategy: str | None = None,
    ) -> None:
        self.trajectory_map = trajectory_map
        self.p_var = str(p_var)
        if self.p_var == "":
            raise ValueError("KotsTrajectoryStateBuilder: p_var must be non-empty.")
        self.jacobian_strategy = _normalize_kots_jacobian_strategy(
            jacobian_strategy,
            prefer_matvec_jacobian=prefer_matvec_jacobian,
        )
        self.trajectory_derivative_maps: dict[int, TrajectoryMap] = {0: trajectory_map}
        if trajectory_derivative_maps is not None:
            for order_raw, traj in trajectory_derivative_maps.items():
                order = int(order_raw)
                if order < 0:
                    raise ValueError(f"KotsTrajectoryStateBuilder: derivative order must be >= 0, got {order}.")
                self.trajectory_derivative_maps[order] = traj
        self._validate_derivative_maps()

        super().__init__(
            model,
            data,
            q_var=self.p_var,
            fields=fields,
            dynamics_fields=dynamics_fields,
            dynamics_owner_type=dynamics_owner_type,
            prefer_matvec_jacobian=prefer_matvec_jacobian,
        )
        self.register_value_and_jac(
            dtype=DTYPE_COORD,
            owner_type="total_joint",
            field="q",
            value_handler=self._handle_joint_q_value,
            jac_handler=self._handle_joint_q_jac,
            jacobian_wrt=STATE_JACOBIAN_VAR,
        )

    def _validate_derivative_maps(self) -> None:
        validate_trajectory_derivative_maps(
            self.trajectory_derivative_maps,
            error_prefix="KotsTrajectoryStateBuilder",
        )

    def _update_trajectory_step(self, *, k: int, q_k: Array, motion_k: Array) -> None:
        del k, q_k
        self._update_kinematics(motion_k)

    def _handle_joint_q_value(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        return np.asarray(q, dtype=float).reshape(-1).copy()

    def _handle_joint_q_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        n = int(np.asarray(q, dtype=float).reshape(-1).size)
        return np.eye(n, dtype=float)

    def _compose_motion_and_jac(self, p: Array, *, k: int) -> tuple[Array, Array]:
        return compose_interleaved_motion_and_jac(
            p,
            trajectory_map=self.trajectory_map,
            trajectory_derivative_maps=self.trajectory_derivative_maps,
            order=self._model_order(),
            k=k,
            error_prefix="KotsTrajectoryStateBuilder",
        )

    def _chain_param_jac(
        self,
        J_raw: Array,
        *,
        key: StateKey,
        jacobian_wrt: str | None,
        dqdp_k: Array,
        dmotiondp_k: Array,
    ) -> Array:
        Jm = np.asarray(J_raw, dtype=float)
        if Jm.ndim != 2:
            raise ValueError(
                f"KotsTrajectoryStateBuilder: Jacobian must be 2D, got shape {Jm.shape} for key {key!r}."
            )

        wrt = None if jacobian_wrt is None else str(jacobian_wrt)
        if wrt == self.q_var:
            return Jm

        if wrt == STATE_JACOBIAN_VAR:
            cols = int(Jm.shape[1])
            extra: tuple[Array, ...] = ()
            dof = self._model_dof()
            order = self._model_order()
            if dof > 0 and cols % dof == 0 and dmotiondp_k.shape[0] == dof * order:
                used_order = int(cols // dof)
                reduced = interleaved_motion_jacobian_used_order(
                    dof=dof,
                    order=order,
                    used_order=used_order,
                    dmotiondp=dmotiondp_k,
                )
                if reduced is not None:
                    extra = (reduced,)
            return chain_param_jacobian(
                Jm,
                q_var=self.q_var,
                state_jacobian_var=STATE_JACOBIAN_VAR,
                key=key,
                jacobian_wrt=jacobian_wrt,
                dqdp=dqdp_k,
                dmotiondp=dmotiondp_k,
                error_prefix="KotsTrajectoryStateBuilder",
                extra_candidates=extra,
            )

        raise ValueError(
            "KotsTrajectoryStateBuilder: unsupported jacobian_wrt metadata for parameter chain. "
            f"Expected {self.q_var!r} or {STATE_JACOBIAN_VAR!r}, got {wrt!r}."
        )

    def _motion_jacobian_used_order_candidate(
        self,
        *,
        used_order: int,
        dmotiondp_k: Array,
    ) -> Array | None:
        dof = self._model_dof()
        order = self._model_order()
        used_order = int(used_order)
        if dof <= 0 or used_order < 1 or used_order > order:
            return None
        if dmotiondp_k.shape[0] != dof * order:
            return None

        cols = int(dof * used_order)
        return interleaved_motion_jacobian_used_order(
            dof=dof,
            order=order,
            used_order=used_order,
            dmotiondp=dmotiondp_k,
        )

    def _preferred_motion_jacobian_used_order(self, key: StateKey | None) -> int | None:
        if key is None or getattr(key, "dtype", None) != DTYPE_DYNAMICS:
            return None
        try:
            field, _var = split_jac_field(str(getattr(key, "field", "")))
        except ValueError:
            field = str(getattr(key, "field", ""))
        field = canonical_field_name(field)
        if field == "torque":
            return min(self._model_order(), 3)
        try:
            return min(self._model_order(), torque_derivative_order(field) + 3)
        except ValueError:
            return None

    def _motion_jacobian_chain_candidates(
        self,
        *,
        dqdp_k: Array,
        dmotiondp_k: Array,
        key: StateKey | None = None,
    ) -> tuple[Array, ...]:
        candidates: list[Array] = []
        preferred_order = self._preferred_motion_jacobian_used_order(key)
        if preferred_order is not None:
            preferred = self._motion_jacobian_used_order_candidate(
                used_order=preferred_order,
                dmotiondp_k=np.asarray(dmotiondp_k, dtype=float),
            )
            if preferred is not None:
                candidates.append(preferred)

        candidates.extend([np.asarray(dmotiondp_k, dtype=float), np.asarray(dqdp_k, dtype=float)])
        dof = self._model_dof()
        order = self._model_order()
        if dof > 0 and dmotiondp_k.shape[0] == dof * order:
            for used_order in range(order, 0, -1):
                dmotion_reduced = self._motion_jacobian_used_order_candidate(
                    used_order=used_order,
                    dmotiondp_k=np.asarray(dmotiondp_k, dtype=float),
                )
                if dmotion_reduced is not None:
                    candidates.append(dmotion_reduced)

        return unique_jacobian_chain_candidates(candidates)

    def _param_jac_from_matvec(
        self,
        *,
        key: StateKey,
        state_ref: Any,
        jacobian_wrt: str | None,
        dqdp_k: Array,
        dmotiondp_k: Array,
    ) -> Array:
        wrt = None if jacobian_wrt is None else str(jacobian_wrt)
        if wrt == self.q_var:
            return self._handle_jac(np.zeros((dqdp_k.shape[0],), dtype=float), key, state_ref)
        if wrt != STATE_JACOBIAN_VAR:
            raise ValueError(
                "KotsTrajectoryStateBuilder: unsupported jacobian_wrt metadata for parameter chain. "
                f"Expected {self.q_var!r} or {STATE_JACOBIAN_VAR!r}, got {wrt!r}."
            )

        total_joint_ref = self._as_total_joint_dynamics_state_ref(state_ref)
        last_error: Exception | None = None
        for cols in self._motion_jacobian_chain_candidates(dqdp_k=dqdp_k, dmotiondp_k=dmotiondp_k, key=key):
            try:
                if total_joint_ref is None:
                    Jp = self._jac_from_matvec_single_state_ref(state_ref, cols)
                    return self._rotate_link_kinematics_jacobian_to_world(J=Jp, key=key, state_ref=state_ref)

                try:
                    return self._jac_from_matvec_single_state_ref(list(total_joint_ref.refs), cols)
                except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
                    pass

                blocks = [self._jac_from_matvec_single_state_ref(ref, cols) for ref in total_joint_ref.refs]
                if len(blocks) == 0:
                    return np.zeros((0, cols.shape[1]), dtype=float)
                ncols = int(blocks[0].shape[1])
                for block in blocks[1:]:
                    if int(block.shape[1]) != ncols:
                        raise ValueError(
                            "KotsTrajectoryStateBuilder: inconsistent matvec column size while stacking "
                            f"total_joint dynamics. Expected {ncols}, got {block.shape[1]}."
                        )
                return np.vstack(blocks)
            except (AttributeError, KeyError, ValueError, TypeError, RuntimeError) as e:
                last_error = e
                continue

        if last_error is not None:
            raise last_error
        raise AttributeError("KotsTrajectoryStateBuilder: model does not expose a usable matvec method.")

    def _should_use_param_jacobian_mul(
        self,
        *,
        key: StateKey,
        state_ref: Any,
        dqdp_k: Array,
        dmotiondp_k: Array,
    ) -> bool:
        del key, state_ref, dqdp_k, dmotiondp_k
        return str(getattr(self, "jacobian_strategy", "mul")) != "dense"

    def _param_jac_transpose_mul_from_state_ref(
        self,
        *,
        key: StateKey,
        state_ref: Any,
        rhs: Array,
        dqdp_k: Array,
        dmotiondp_k: Array,
    ) -> Array:
        rhs_local = self._rotate_link_kinematics_rhs_to_local(
            rhs=np.asarray(rhs, dtype=float),
            key=key,
            state_ref=state_ref,
        )
        motion_grad = np.asarray(self._transpose_matvec_from_state_ref(state_ref, rhs_local), dtype=float)

        if motion_grad.ndim == 1:
            for cols in self._motion_jacobian_chain_candidates(dqdp_k=dqdp_k, dmotiondp_k=dmotiondp_k, key=key):
                if int(cols.shape[0]) == int(motion_grad.size):
                    return np.asarray(cols.T @ motion_grad.reshape(-1), dtype=float).reshape(-1)
        elif motion_grad.ndim == 2:
            for cols in self._motion_jacobian_chain_candidates(dqdp_k=dqdp_k, dmotiondp_k=dmotiondp_k, key=key):
                if int(cols.shape[0]) == int(motion_grad.shape[0]):
                    return np.asarray(cols.T @ motion_grad, dtype=float)
        else:
            raise ValueError(
                "KotsTrajectoryStateBuilder: jacobian_transpose_mul output must be 1D or 2D, "
                f"got shape {motion_grad.shape}."
            )

        raise ValueError(
            "KotsTrajectoryStateBuilder: transpose Jacobian chain mismatch. "
            f"transpose output has shape {motion_grad.shape}, dqdp has shape {dqdp_k.shape}, "
            f"dmotiondp has shape {dmotiondp_k.shape}."
        )

    def param_jacobian_transpose_mul(
        self,
        x_all: Array,
        key: StateKey,
        rhs: Array,
        *,
        pack: Any = None,
        time: Any = None,
        update_kinematics: bool = True,
    ) -> Array:
        """Compute `J_p(key).T @ rhs` for a trajectory state key without forming `J_p`."""

        steps = self._expected_steps(time=time)
        if not self._accept_required_key_for_traj(key, steps=steps):
            raise ValueError(
                "KotsTrajectoryStateBuilder: invalid state key for param_jacobian_transpose_mul: "
                f"{key!r}"
            )
        route = self._route_for_key(key)
        entry = None if route is None else self._dispatch.get(route, None)
        if entry is None:
            raise ValueError(f"KotsTrajectoryStateBuilder: no value handler route for key: {key!r}")

        p = self._extract_q(np.asarray(x_all, dtype=float).reshape(-1), pack=pack)
        if p.size != self.trajectory_map.p_dim:
            raise ValueError(
                "KotsTrajectoryStateBuilder: parameter size mismatch. "
                f"Expected p_dim={self.trajectory_map.p_dim}, got {p.size}."
            )

        k = int(key.k)
        dqdp_k = self.trajectory_map.dqdp_at(k)
        motion_k, dmotiondp_k = self._compose_motion_and_jac(p, k=k)
        if update_kinematics:
            self._update_kinematics(motion_k)
        state_ref = self._state_ref(key, state_ref_field=entry.state_ref_field)
        return self._param_jac_transpose_mul_from_state_ref(
            key=key,
            state_ref=state_ref,
            rhs=rhs,
            dqdp_k=dqdp_k,
            dmotiondp_k=dmotiondp_k,
        )

    def _evaluate_trajectory_entry(
        self,
        *,
        key: StateKey,
        entry: Any,
        state_ref: Any,
        q_k: Array,
        dqdp_k: Array,
        dmotiondp_k: Array,
    ) -> Any:
        is_param_jac = self._is_param_jac_key(key)
        if is_param_jac and self._should_use_param_jacobian_mul(
            key=key,
            state_ref=state_ref,
            dqdp_k=dqdp_k,
            dmotiondp_k=dmotiondp_k,
        ):
            try:
                return self._param_jac_from_matvec(
                    key=key,
                    state_ref=state_ref,
                    jacobian_wrt=entry.jacobian_wrt,
                    dqdp_k=dqdp_k,
                    dmotiondp_k=dmotiondp_k,
                )
            except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
                pass
        return super()._evaluate_trajectory_entry(
            key=key,
            entry=entry,
            state_ref=state_ref,
            q_k=q_k,
            dqdp_k=dqdp_k,
            dmotiondp_k=dmotiondp_k,
        )


__all__ = [
    "StateType",
    "KotsFieldFamily",
    "KOTS_DEFAULT_FIELD_FAMILIES",
    "KotsStateBuilder",
    "KotsTrajectoryStateBuilder",
]
