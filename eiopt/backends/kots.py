from __future__ import annotations

from collections.abc import Mapping, Sequence, Iterable
from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from ..core.state_cache import StateKey
from ..core.state_schema import (
    DYNAMICS_FIELDS,
    DTYPE_COORD,
    DTYPE_DYNAMICS,
    DTYPE_KINEMATICS,
    canonical_field_name,
    split_jac_field,
    torque_derivative_field,
    torque_derivative_order,
)
from ..core.trajectory import TrajectoryMap
from ..dsl import compile_problem
from ..dsl.dsl_ops import find_var_dsl
from ..dsl.trajectory import (
    build_trajectory_map,
    build_trajectory_maps_with_derivatives,
    default_dt_from_time,
    default_steps_from_time,
)
from ..model.runtime import ProblemRuntime
from ._template import BackendDispatchStateBuilder

try:
    from robokots.core.state import StateType
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "`eiopt.backends.kots` requires the robotics RoboKots bindings. "
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


@dataclass(frozen=True)
class KotsTrajectoryCompiledProblem:
    runtime: ProblemRuntime
    builder: "KotsTrajectoryStateBuilder"
    trajectory_map: TrajectoryMap
    trajectory_derivative_maps: dict[int, TrajectoryMap]
    p_var: str
    dt: float
    model_order: int
    dynamics_fields: tuple[str, ...] = ()


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
        return StateType(str(owner_type), owner_name, state_field, str(frame_name))

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
            refs.append(StateType("joint", joint_name, state_field, frame_name))
        return tuple(refs)

    @staticmethod
    def _state_field_name(field: Any) -> str:
        field_name = canonical_field_name(str(field))
        order = torque_derivative_order(field_name)
        if order is None:
            return field_name
        if order == 0:
            return field_name
        return f"torque_diff{order}"

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
        m = None if data_type is None else _ROBOKOTS_TORQUE_DIFF_PATTERN.fullmatch(data_type)
        if m is not None:
            diff_order = int(m.group(1))
            eiopt_field = torque_derivative_field(diff_order)
            required_model_order = diff_order + 3
            raise ValueError(
                "KotsStateBuilder: dynamics field "
                f"{eiopt_field!r} requires RoboKots model order >= {required_model_order}. "
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


def _mapping_as_dict(mapping: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    if isinstance(mapping, dict):
        return mapping
    try:
        return dict(mapping)
    except Exception as e:
        raise TypeError(f"{where} must be a mapping.") from e


def _infer_model_dof(model: Any) -> int | None:
    dof_fn = getattr(model, "dof", None)
    if callable(dof_fn):
        try:
            return int(dof_fn())
        except Exception:
            return None
    robot = getattr(model, "robot_", None)
    if robot is not None and hasattr(robot, "dof"):
        try:
            return int(getattr(robot, "dof"))
        except Exception:
            return None
    return None


def _infer_model_order(model: Any) -> int:
    order_fn = getattr(model, "order", None)
    if callable(order_fn):
        try:
            return max(1, int(order_fn()))
        except Exception:
            pass

    order_attr = getattr(model, "order_", None)
    if order_attr is None:
        return 1
    try:
        return max(1, int(order_attr))
    except Exception:
        return 1


def _resolve_dt(dsl: Mapping[str, Any], *, default_dt: float | None = None) -> float:
    dt = default_dt_from_time(dsl)
    if dt is None:
        dt = default_dt
    if dt is None:
        dt = 1.0
    dt_f = float(dt)
    if dt_f <= 0.0:
        raise ValueError(f"time.dt must be > 0. Got {dt_f}.")
    return dt_f


def _resolve_p_var_name(*, dsl: Mapping[str, Any], trajectory_dsl: Mapping[str, Any], p_var: str | None) -> str:
    if p_var is not None:
        name = str(p_var).strip()
    else:
        name = str(trajectory_dsl.get("var", "p")).strip()
    if name == "":
        raise ValueError("trajectory.var must be non-empty.")

    var_dsl = find_var_dsl(_mapping_as_dict(dsl, where="dsl"), name=name)
    if var_dsl is None:
        raise ValueError(f"DSL must declare variable {name!r}.")
    return name


def _base_field_name(field: str) -> str:
    field_name = canonical_field_name(str(field))
    try:
        base, _var = split_jac_field(field_name)
    except ValueError:
        return field_name
    return base


def _canonicalize_dynamics_fields(
    dynamics_fields: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if dynamics_fields is None:
        return None
    out: list[str] = []
    seen: set[str] = set()
    for field_raw in dynamics_fields:
        field = canonical_field_name(str(field_raw).strip())
        if field == "":
            continue
        if field in seen:
            continue
        seen.add(field)
        out.append(field)
    if len(out) == 0:
        raise ValueError("compile_kots_trajectory_problem: dynamics_fields must be non-empty when provided.")
    return tuple(out)


def _registered_dynamics_base_fields(
    *,
    builder: KotsTrajectoryStateBuilder,
    owner_type: str,
) -> set[str]:
    fields: set[str] = set()
    for dtype, route_owner_type, field in builder._dispatch.keys():
        if dtype != DTYPE_DYNAMICS or route_owner_type != owner_type:
            continue
        fields.add(_base_field_name(field))
    return fields


def _required_dynamics_base_fields_in_order(
    *,
    runtime: ProblemRuntime,
    owner_type: str,
) -> tuple[list[str], set[str]]:
    requested_fields: list[str] = []
    requested_seen: set[str] = set()
    unsupported_owner_types: set[str] = set()
    for key in runtime.required:
        if getattr(key, "dtype", None) != DTYPE_DYNAMICS:
            continue
        owner = getattr(key, "owner", None)
        key_owner_type = getattr(owner, "owner_type", None)
        if key_owner_type != owner_type:
            unsupported_owner_types.add(str(key_owner_type))
            continue
        field = _base_field_name(str(getattr(key, "field", "")))
        if field in requested_seen:
            continue
        requested_seen.add(field)
        requested_fields.append(field)
    return requested_fields, unsupported_owner_types


def _required_dynamics_base_fields(
    *,
    runtime: ProblemRuntime,
    owner_type: str,
) -> tuple[set[str], set[str]]:
    requested_fields, unsupported_owner_types = _required_dynamics_base_fields_in_order(
        runtime=runtime,
        owner_type=owner_type,
    )
    return set(requested_fields), unsupported_owner_types


def _validate_model_order_for_dynamics_fields(
    *,
    model_order: int,
    dynamics_fields: Sequence[str] | None,
) -> None:
    if dynamics_fields is None:
        return
    for field in dynamics_fields:
        deriv_order = torque_derivative_order(str(field))
        if deriv_order is None or deriv_order <= 0:
            continue
        required_model_order = int(deriv_order) + 3
        if int(model_order) >= required_model_order:
            continue
        raise ValueError(
            "compile_kots_trajectory_problem: dynamics field "
            f"{field!r} requires RoboKots model order >= {required_model_order}. "
            f"Current model order is {int(model_order)}."
        )


def _validate_kots_runtime_dynamics_coverage(
    *,
    runtime: ProblemRuntime,
    builder: KotsTrajectoryStateBuilder,
    dynamics_owner_type: str,
) -> None:
    requested_fields, unsupported_owner_types = _required_dynamics_base_fields(
        runtime=runtime,
        owner_type=dynamics_owner_type,
    )
    if unsupported_owner_types:
        unsupported = ", ".join(sorted(unsupported_owner_types))
        raise ValueError(
            "compile_kots_trajectory_problem: DSL contains dynamics keys with unsupported owner_type(s): "
            f"{unsupported}. Supported owner_type is {dynamics_owner_type!r}."
        )
    if len(requested_fields) == 0:
        return

    registered_fields = _registered_dynamics_base_fields(
        builder=builder,
        owner_type=dynamics_owner_type,
    )
    missing_fields = sorted(requested_fields - registered_fields)
    if len(missing_fields) == 0:
        return

    requested_str = ", ".join(sorted(requested_fields))
    registered_str = ", ".join(sorted(registered_fields)) if len(registered_fields) > 0 else "<none>"
    missing_str = ", ".join(missing_fields)
    raise ValueError(
        "compile_kots_trajectory_problem: DSL requests dynamics field(s) that are not registered in "
        "KotsTrajectoryStateBuilder. "
        f"Missing: {missing_str}. Requested: {requested_str}. Registered: {registered_str}. "
        "Add missing entries to `dynamics_fields` (e.g. include 'torque_d1' for first torque derivative), "
        "or remove corresponding get_state dynamics terms."
    )


def compile_kots_trajectory_problem(
    dsl: Mapping[str, Any],
    *,
    model: Any,
    data: Any,
    p_var: str | None = None,
    max_derivative_order: int | None = None,
    derivative_wrt: str = "time",
    default_steps: int | None = None,
    default_q_dim: int | None = None,
    default_dt: float | None = None,
    fields: Sequence[str] | None = None,
    dynamics_fields: Sequence[str] | None = None,
    dynamics_owner_type: str = "total_joint",
) -> KotsTrajectoryCompiledProblem:
    """Compile a trajectory-parameterized Kots optimization runtime from DSL.

    This helper bundles:
      1) trajectory map construction
      2) derivative trajectory map construction
      3) trajectory variable dimension validation
      4) KotsTrajectoryStateBuilder setup
      5) compile_problem(..., build_state=builder.build_state)
    """

    dsl_dict = _mapping_as_dict(dsl, where="dsl")
    trajectory_dsl_raw = dsl_dict.get("trajectory", None)
    if not isinstance(trajectory_dsl_raw, Mapping):
        raise ValueError("DSL must contain [trajectory] section.")
    trajectory_dsl = _mapping_as_dict(trajectory_dsl_raw, where="dsl.trajectory")

    p_var_name = _resolve_p_var_name(dsl=dsl_dict, trajectory_dsl=trajectory_dsl, p_var=p_var)

    if default_steps is None:
        default_steps = default_steps_from_time(dsl_dict)
    if default_q_dim is None:
        default_q_dim = _infer_model_dof(model)

    traj_map = build_trajectory_map(
        trajectory_dsl,
        default_steps=default_steps,
        default_q_dim=default_q_dim,
    )
    dt = _resolve_dt(dsl_dict, default_dt=default_dt)
    model_order = _infer_model_order(model)

    if max_derivative_order is None:
        max_derivative_order_use = max(0, model_order - 1)
    else:
        max_derivative_order_use = int(max_derivative_order)
        if max_derivative_order_use < 0:
            raise ValueError(
                f"max_derivative_order must be >= 0, got {max_derivative_order_use}."
            )

    traj_maps = build_trajectory_maps_with_derivatives(
        trajectory_dsl,
        max_derivative_order=max_derivative_order_use,
        derivative_wrt=derivative_wrt,
        default_steps=traj_map.steps,
        default_q_dim=traj_map.q_dim,
        default_dt=dt,
    )
    traj_maps_by_order = {i: m for i, m in enumerate(traj_maps)}

    p_var_dsl = find_var_dsl(dsl_dict, name=p_var_name)
    if p_var_dsl is None:
        raise ValueError(f"DSL must declare variable {p_var_name!r}.")
    p_dim_dsl = int(p_var_dsl.get("dim", -1))
    if p_dim_dsl != traj_map.p_dim:
        raise ValueError(
            f"variable {p_var_name!r} dim mismatch: dsl={p_dim_dsl}, trajectory p_dim={traj_map.p_dim}."
        )

    dynamics_fields_use = _canonicalize_dynamics_fields(dynamics_fields)
    if dynamics_fields_use is None:
        probe_builder = KotsTrajectoryStateBuilder(
            model,
            data,
            trajectory_map=traj_map,
            trajectory_derivative_maps=traj_maps_by_order,
            p_var=p_var_name,
            fields=fields,
            dynamics_fields=None,
            dynamics_owner_type=dynamics_owner_type,
        )
        probe_runtime = compile_problem(dsl_dict, build_state=probe_builder.build_state)
        requested_fields_order, unsupported_owner_types = _required_dynamics_base_fields_in_order(
            runtime=probe_runtime,
            owner_type=dynamics_owner_type,
        )
        if unsupported_owner_types:
            unsupported = ", ".join(sorted(unsupported_owner_types))
            raise ValueError(
                "compile_kots_trajectory_problem: DSL contains dynamics keys with unsupported owner_type(s): "
                f"{unsupported}. Supported owner_type is {dynamics_owner_type!r}."
            )
        dynamics_fields_use = tuple(requested_fields_order) if len(requested_fields_order) > 0 else None

    _validate_model_order_for_dynamics_fields(
        model_order=int(model_order),
        dynamics_fields=dynamics_fields_use,
    )

    builder = KotsTrajectoryStateBuilder(
        model,
        data,
        trajectory_map=traj_map,
        trajectory_derivative_maps=traj_maps_by_order,
        p_var=p_var_name,
        fields=fields,
        dynamics_fields=dynamics_fields_use,
        dynamics_owner_type=dynamics_owner_type,
    )
    runtime = compile_problem(dsl_dict, build_state=builder.build_state)
    _validate_kots_runtime_dynamics_coverage(
        runtime=runtime,
        builder=builder,
        dynamics_owner_type=dynamics_owner_type,
    )

    return KotsTrajectoryCompiledProblem(
        runtime=runtime,
        builder=builder,
        trajectory_map=traj_map,
        trajectory_derivative_maps=traj_maps_by_order,
        p_var=p_var_name,
        dt=float(dt),
        model_order=int(model_order),
        dynamics_fields=tuple() if dynamics_fields_use is None else tuple(dynamics_fields_use),
    )
