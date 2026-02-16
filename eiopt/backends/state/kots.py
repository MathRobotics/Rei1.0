from __future__ import annotations

from collections.abc import Mapping, Sequence, Iterable
from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from ...core.state_cache import StateKey
from ...core.state_schema import (
    DYNAMICS_FIELDS,
    DTYPE_COORD,
    DTYPE_DYNAMICS,
    DTYPE_KINEMATICS,
    canonical_field_name,
    split_jac_field,
    torque_derivative_field,
    torque_derivative_order,
)
from ...core.trajectory import TrajectoryMap
from .template import BackendDispatchStateBuilder

try:
    from robokots.core.state import StateType
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "`eiopt.backends.state.kots` requires the robotics RoboKots bindings. "
        "Install RoboKots (e.g. via github) and retry."
    ) from e

Array = np.ndarray
STATE_JACOBIAN_VAR = "state"
_ROBOKOTS_TORQUE_DIFF_PATTERN = re.compile(r"^torque_diff([1-9][0-9]*)$")


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
    ) -> None:
        super().__init__(model, data, q_var=q_var)
        self.dtype = DTYPE_KINEMATICS
        self.owner_type = "link"
        self.dynamics_owner_type = str(dynamics_owner_type)
        if self.dynamics_owner_type == "":
            raise ValueError("KotsStateBuilder: dynamics_owner_type must be non-empty.")
        self._needs_dynamics_update = False

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

    def _update_dynamics_if_available(self) -> None:
        if not self._needs_dynamics_update:
            return
        for name in ("dynamics", "compute_dynamics", "update_dynamics"):
            fn = getattr(self.model, name, None)
            if not callable(fn):
                continue
            try:
                fn()
            except TypeError:
                continue
            return

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
        self.model.kinematics()
        self._update_dynamics_if_available()

    def _model_dof(self) -> int:
        dof_fn = getattr(self.model, "dof", None)
        if callable(dof_fn):
            return int(dof_fn())
        robot = getattr(self.model, "robot_", None)
        if robot is not None and hasattr(robot, "dof"):
            return int(getattr(robot, "dof"))
        raise ValueError("KotsStateBuilder: unable to resolve model dof.")

    def _model_order(self) -> int:
        order_fn = getattr(self.model, "order", None)
        if callable(order_fn):
            order = int(order_fn())
        else:
            order = int(getattr(self.model, "order_", 1))
        if order < 1:
            raise ValueError(f"KotsStateBuilder: model order must be >= 1, got {order}.")
        return order

    def _expand_coordinate_motion(self, q: Array, *, dof: int, order: int) -> Array:
        motion = np.zeros(dof * order, dtype=float)
        robot = getattr(self.model, "robot_", None)
        if robot is None:
            for i in range(min(q.size, dof)):
                motion[i * order] = float(q[i])
            return motion

        owners = [*getattr(robot, "links", []), *getattr(robot, "joints", [])]
        owners = [owner for owner in owners if int(getattr(owner, "dof", 0)) > 0]
        owners.sort(key=lambda owner: int(getattr(owner, "dof_index", 0)))

        cursor = 0
        for owner in owners:
            owner_dof = int(getattr(owner, "dof", 0))
            dof_index = int(getattr(owner, "dof_index", 0))
            start = dof_index * order
            stop = start + owner_dof
            if stop > motion.size:
                raise ValueError("KotsStateBuilder: invalid dof_index/dof in robot structure.")
            motion[start:stop] = q[cursor : cursor + owner_dof]
            cursor += owner_dof

        if cursor != q.size:
            raise ValueError(
                "KotsStateBuilder: failed to map q into motion coordinates. "
                f"Mapped {cursor} elements from q size {q.size}."
            )
        return motion

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
        deriv_order = torque_derivative_order(state_field)
        if isinstance(deriv_order, int) and deriv_order > 0:
            return f"torque_diff{deriv_order}"
        return state_field

    def _make_state_type(
        self,
        *,
        owner_type: str,
        owner_name: str,
        state_field: str,
        frame_name: str | None,
    ) -> Any:
        try:
            return StateType(owner_type, owner_name, state_field, frame_name)
        except KeyError:
            fallback_field = self._fallback_backend_field_name(state_field)
            if fallback_field == state_field:
                raise
            return StateType(owner_type, owner_name, fallback_field, frame_name)

    @staticmethod
    def _torque_derivative_order_from_state_data_type(data_type: str) -> int | None:
        try:
            deriv_order = torque_derivative_order(data_type)
        except ValueError:
            deriv_order = None
        if isinstance(deriv_order, int):
            return deriv_order
        m = _ROBOKOTS_TORQUE_DIFF_PATTERN.fullmatch(data_type)
        if m is None:
            return None
        return int(m.group(1))

    def _dof_sorted_joints(self) -> list[Any] | None:
        robot = getattr(self.model, "robot_", None)
        if robot is None:
            return None
        joints_raw = getattr(robot, "joints", None)
        if joints_raw is None:
            return None
        joints = [joint for joint in joints_raw if int(getattr(joint, "dof", 0)) > 0]
        joints.sort(key=lambda joint: int(getattr(joint, "dof_index", 0)))
        return joints

    def _total_joint_dynamics_frame_name(self, *, state_field: str, key: StateKey) -> str | None:
        frame_name = getattr(key, "frame", None) or "world"
        # torque family does not need world-frame conversion in RoboKots get_value path.
        # Keeping frame unset avoids a known owner_type='total_joint' world-path bug upstream.
        if "torque" in state_field:
            return None
        return str(frame_name)

    @staticmethod
    def _state_ref_data_type(state_ref: Any) -> str | None:
        for attr in ("data_type", "field", "field_", "dtype"):
            value = getattr(state_ref, attr, None)
            if isinstance(value, str) and value != "":
                return value
        return None

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
            return np.asarray(self.model.state_info(state_ref), dtype=float).reshape(-1)
        except KeyError as e:
            self._raise_missing_state_key(state_ref=state_ref, cause=e)

    def _value_from_state_ref(self, state_ref: Any) -> Array:
        total_joint_ref = self._as_total_joint_dynamics_state_ref(state_ref)
        if total_joint_ref is not None:
            return self._concat_total_joint_values(total_joint_ref.refs)

        return self._value_from_single_state_ref(state_ref)

    def _jac_from_single_state_ref(self, state_ref: Any) -> Array:
        try:
            J = np.asarray(self.model.jacobian(state_ref), dtype=float)
        except KeyError as e:
            self._raise_missing_state_key(state_ref=state_ref, cause=e)
        if J.ndim != 2:
            raise ValueError(f"Kots Jacobian must be 2D, got shape {J.shape}.")

        m = int(self._value_from_single_state_ref(state_ref).size)
        if J.shape[0] == m:
            return J
        if J.shape[1] == m:
            return J.T
        raise ValueError(f"Kots Jacobian must be ({m},n) or (n,{m}), got {J.shape}.")

    def _handle_value(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del q, key
        return self._value_from_state_ref(state_ref)

    def _handle_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key
        del q
        total_joint_ref = self._as_total_joint_dynamics_state_ref(state_ref)
        if total_joint_ref is not None:
            return self._stack_total_joint_jacobians(total_joint_ref.refs)

        return self._jac_from_single_state_ref(state_ref)

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


class KotsTrajectoryStateBuilder(KotsStateBuilder):
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
    ) -> None:
        self.trajectory_map = trajectory_map
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
            q_var=p_var,
            fields=fields,
            dynamics_fields=dynamics_fields,
            dynamics_owner_type=dynamics_owner_type,
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
        base = self.trajectory_derivative_maps.get(0, None)
        if base is None:
            raise ValueError("KotsTrajectoryStateBuilder: derivative map for order 0 is required.")
        for order, traj in self.trajectory_derivative_maps.items():
            if traj.p_dim != base.p_dim:
                raise ValueError(
                    "KotsTrajectoryStateBuilder: derivative map p_dim mismatch. "
                    f"order={order}, expected {base.p_dim}, got {traj.p_dim}."
                )
            if traj.steps != base.steps or traj.q_dim != base.q_dim:
                raise ValueError(
                    "KotsTrajectoryStateBuilder: derivative map shape mismatch. "
                    f"order={order}, expected steps={base.steps}, q_dim={base.q_dim}, "
                    f"got steps={traj.steps}, q_dim={traj.q_dim}."
                )

    def _handle_joint_q_value(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        return np.asarray(q, dtype=float).reshape(-1).copy()

    def _handle_joint_q_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        n = int(np.asarray(q, dtype=float).reshape(-1).size)
        return np.eye(n, dtype=float)

    def _compose_motion_and_jac(self, p: Array, *, k: int) -> tuple[Array, Array]:
        p_vec = np.asarray(p, dtype=float).reshape(-1)
        dof = int(self.trajectory_map.q_dim)
        order = self._model_order()
        motion = np.zeros((dof * order,), dtype=float)
        dmotion_dp = np.zeros((dof * order, int(self.trajectory_map.p_dim)), dtype=float)

        for deriv_order, traj in self.trajectory_derivative_maps.items():
            if deriv_order >= order:
                continue
            q_r = np.asarray(traj.q_at(p_vec, k), dtype=float).reshape(-1)
            J_r = np.asarray(traj.dqdp_at(k), dtype=float)
            if q_r.size != dof:
                raise ValueError(
                    "KotsTrajectoryStateBuilder: derivative map q size mismatch. "
                    f"order={deriv_order}, expected {dof}, got {q_r.size}."
                )
            if J_r.shape != (dof, self.trajectory_map.p_dim):
                raise ValueError(
                    "KotsTrajectoryStateBuilder: derivative map jacobian shape mismatch. "
                    f"order={deriv_order}, expected {(dof, self.trajectory_map.p_dim)}, got {J_r.shape}."
                )
            motion[deriv_order::order] = q_r
            dmotion_dp[deriv_order::order, :] = J_r

        return motion, dmotion_dp

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
            if Jm.shape[1] == dqdp_k.shape[0]:
                return Jm @ dqdp_k
            if Jm.shape[1] == dmotiondp_k.shape[0]:
                return Jm @ dmotiondp_k
            # Some RoboKots dynamics Jacobians only depend on lower motion orders
            # even when model.order is higher (e.g. torque at order=4 still uses q,dq,ddq).
            dof = self._model_dof()
            order = self._model_order()
            cols = int(Jm.shape[1])
            if dof > 0 and cols % dof == 0 and dmotiondp_k.shape[0] == dof * order:
                used_order = int(cols // dof)
                if 1 <= used_order <= order:
                    dmotion_reduced = np.zeros((cols, dmotiondp_k.shape[1]), dtype=float)
                    for i in range(dof):
                        src0 = i * order
                        src1 = src0 + used_order
                        dst0 = i * used_order
                        dst1 = dst0 + used_order
                        dmotion_reduced[dst0:dst1, :] = dmotiondp_k[src0:src1, :]
                    return Jm @ dmotion_reduced
            raise ValueError(
                "KotsTrajectoryStateBuilder: Jacobian chain mismatch. "
                f"J_raw has shape {Jm.shape}, dqdp has shape {dqdp_k.shape}, dmotiondp has shape {dmotiondp_k.shape}."
            )

        raise ValueError(
            "KotsTrajectoryStateBuilder: unsupported jacobian_wrt metadata for parameter chain. "
            f"Expected {self.q_var!r} or {STATE_JACOBIAN_VAR!r}, got {wrt!r}."
        )

    def _expected_steps(self, *, time: Any = None) -> int:
        steps = int(self.trajectory_map.steps)
        if time is None or not hasattr(time, "N"):
            return steps
        try:
            time_steps = int(time.N) + 1
        except Exception:
            return steps
        if time_steps != steps:
            raise ValueError(
                "KotsTrajectoryStateBuilder: time grid mismatch. "
                f"trajectory_map.steps={steps}, time steps={time_steps} (N+1)."
            )
        return steps

    def _accept_required_key_for_traj(self, key: StateKey, *, steps: int) -> bool:
        if not isinstance(key, StateKey):
            return False
        k = int(getattr(key, "k", -1))
        if k < 0 or k >= steps:
            return False
        dtype = getattr(key, "dtype", None)
        if not isinstance(dtype, str) or dtype == "":
            return False
        owner = getattr(key, "owner", None)
        owner_type = getattr(owner, "owner_type", None)
        owner_name = getattr(owner, "owner_name", None)
        if not isinstance(owner_type, str) or owner_type == "":
            return False
        if not isinstance(owner_name, str) or owner_name == "":
            return False
        field = getattr(key, "field", None)
        if not isinstance(field, str) or field == "":
            return False
        return True

    def _is_param_jac_key(self, key: StateKey) -> bool:
        field = getattr(key, "field", None)
        if not isinstance(field, str) or field == "":
            return False
        try:
            _base, var = split_jac_field(field)
        except ValueError:
            return False
        return var == self.q_var

    def build_state(
        self,
        x_all: Array,
        *,
        pack: Any = None,
        time: Any = None,
        required: Iterable[StateKey] | None = None,
    ) -> dict[StateKey, Any]:
        if required is None:
            return {}

        steps = self._expected_steps(time=time)
        p = self._extract_q(x_all, pack=pack)
        if p.size != self.trajectory_map.p_dim:
            raise ValueError(
                "KotsTrajectoryStateBuilder: parameter size mismatch. "
                f"Expected p_dim={self.trajectory_map.p_dim}, got {p.size}."
            )

        grouped: dict[int, list[tuple[StateKey, Any]]] = {}
        for key in required:
            if not self._accept_required_key_for_traj(key, steps=steps):
                continue
            route = self._route_for_key(key)
            if route is None:
                continue
            entry = self._dispatch.get(route, None)
            if entry is None:
                continue
            grouped.setdefault(int(key.k), []).append((key, entry))

        out: dict[StateKey, Any] = {}
        for k in sorted(grouped.keys()):
            q_k = self.trajectory_map.q_at(p, k)
            dqdp_k = self.trajectory_map.dqdp_at(k)
            motion_k, dmotiondp_k = self._compose_motion_and_jac(p, k=k)
            self._update_kinematics(motion_k)

            for key, entry in grouped[k]:
                state_ref = self._state_ref(key, state_ref_field=entry.state_ref_field)
                value = entry.handler(q_k, key, state_ref)

                if self._is_param_jac_key(key):
                    value = self._chain_param_jac(
                        value,
                        key=key,
                        jacobian_wrt=entry.jacobian_wrt,
                        dqdp_k=dqdp_k,
                        dmotiondp_k=dmotiondp_k,
                    )

                out[key] = value

        return out


__all__ = [
    "StateType",
    "KotsFieldFamily",
    "KOTS_DEFAULT_FIELD_FAMILIES",
    "KotsStateBuilder",
    "KotsTrajectoryStateBuilder",
]
