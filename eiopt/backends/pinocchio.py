from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.state_cache import StateKey
from ..core.state_schema import (
    DTYPE_COORD,
    DTYPE_DYNAMICS,
    DTYPE_KINEMATICS,
    canonical_field_name,
    split_jac_field,
)
from ..core.trajectory import TrajectoryMap
from ..dsl import compile_problem
from ..dsl.trajectory_compile import PreparedTrajectoryProblemDsl
from ..model.runtime import ProblemRuntime
from ._template import BackendDispatchStateBuilder
from ._spatial import Jacobian6Order, linear_part_from_jacobian6
from .trajectory_adapter import compile_trajectory_problem_with_adapter

try:
    import pinocchio as pin
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "`eiopt.backends.pinocchio` requires the robotics Pinocchio bindings. "
        "Install Pinocchio (e.g. via conda-forge) and retry."
    ) from e

_REQUIRED_ATTRS = ("buildModelFromUrdf", "computeFrameJacobian", "forwardKinematics", "updateFramePlacements")
_missing = [a for a in _REQUIRED_ATTRS if not hasattr(pin, a)]
if _missing:  # pragma: no cover
    raise ImportError(
        "The imported `pinocchio` module is missing required robotics APIs: "
        f"{', '.join(_missing)}.\n"
        "You may have installed a different PyPI package named `pinocchio` (often version 0.1).\n"
        "Uninstall it and install the robotics Pinocchio bindings (typically via conda-forge)."
    )

Array = np.ndarray
STATE_JACOBIAN_VAR = "state"


def compute_pinocchio_frame_jacobian(model: Any, data: Any, q: Array, frame_ref: Any) -> Array:
    """Pinocchio-specific `computeFrameJacobian` wrapper with API/version fallbacks."""
    frame_id = int(frame_ref)

    ref = getattr(pin, "ReferenceFrame", None)
    for name in ("LOCAL_WORLD_ALIGNED", "LOCAL"):
        rf = getattr(ref, name, None) if ref is not None else None
        if rf is None:
            continue
        try:
            return pin.computeFrameJacobian(model, data, q, frame_id, rf)
        except TypeError:
            break

    return pin.computeFrameJacobian(model, data, q, frame_id)


@dataclass(frozen=True)
class PinocchioFieldFamily:
    field: str
    value_handler_name: str
    jac_handler_name: str


@dataclass(frozen=True)
class PinocchioTrajectoryCompiledProblem:
    runtime: ProblemRuntime
    trajectory_map: TrajectoryMap
    trajectory_derivative_maps: dict[int, TrajectoryMap]
    p_var: str
    dt: float
    model_order: int
    dynamics_fields: tuple[str, ...] = ()


# pinocchio.py 内で「どの field ファミリを提供するか」を宣言する登録リスト。
# StateKey の (dtype, owner_type, field) マッチは BackendDispatchStateBuilder 側で自動実行される。
PINOCCHIO_DEFAULT_FIELD_FAMILIES: tuple[PinocchioFieldFamily, ...] = (
    PinocchioFieldFamily(field="pos", value_handler_name="_handle_pos", jac_handler_name="_handle_pos_jac"),
    PinocchioFieldFamily(field="rot", value_handler_name="_handle_rot", jac_handler_name="_handle_rot_jac"),
    PinocchioFieldFamily(field="frame", value_handler_name="_handle_frame", jac_handler_name="_handle_frame_jac"),
)


class PinocchioStateBuilder(BackendDispatchStateBuilder):
    """Pinocchio -> `build_state()` bridge with StateKey-based automatic dispatch.

    `PINOCCHIO_DEFAULT_FIELD_FAMILIES` をもとに、
    `(dtype="kinematics", owner_type="link", field=<...>)` の handler を一括登録する。
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        q_var: str = "q",
        jac6_order: Jacobian6Order = "linear_angular",
        finite_diff_eps: float = 1e-8,
        fields: Sequence[str] | None = None,
        dynamics_fields: Sequence[str] | None = None,
        dynamics_owner_type: str = "total_joint",
        dynamics_custom_handlers: Mapping[str, tuple[Callable[..., Array], Callable[..., Array]]] | None = None,
    ) -> None:
        super().__init__(model, data, q_var=q_var)
        self.dtype = DTYPE_KINEMATICS
        self.owner_type = "link"
        self.dynamics_owner_type = str(dynamics_owner_type)
        if self.dynamics_owner_type == "":
            raise ValueError("PinocchioStateBuilder: dynamics_owner_type must be non-empty.")
        self.jac6_order = jac6_order
        self.finite_diff_eps = float(finite_diff_eps)
        if self.finite_diff_eps <= 0.0:
            raise ValueError("PinocchioStateBuilder: finite_diff_eps must be > 0.")

        family_map = {spec.field: spec for spec in PINOCCHIO_DEFAULT_FIELD_FAMILIES}
        selected_fields = [spec.field for spec in PINOCCHIO_DEFAULT_FIELD_FAMILIES] if fields is None else [str(f) for f in fields]
        if len(selected_fields) == 0:
            raise ValueError("PinocchioStateBuilder: fields must be non-empty.")

        self.field_to_jac: dict[str, str] = {}
        for field in selected_fields:
            spec = family_map.get(field, None)
            if spec is None:
                supported = ", ".join(sorted(family_map.keys()))
                raise ValueError(
                    f"PinocchioStateBuilder: unsupported field {field!r}. "
                    f"Supported fields: {supported}."
                )
            value_handler = getattr(self, spec.value_handler_name)
            jac_handler = getattr(self, spec.jac_handler_name)
            _value_name, jac_name = self.register_value_and_jac(
                dtype=self.dtype,
                owner_type=self.owner_type,
                field=spec.field,
                value_handler=value_handler,
                jac_handler=jac_handler,
                jacobian_wrt=STATE_JACOBIAN_VAR,
            )
            self.field_to_jac[spec.field] = jac_name

        if dynamics_fields is not None:
            dyn_fields = [canonical_field_name(str(f)) for f in dynamics_fields]
            dyn_fields = list(dict.fromkeys(dyn_fields))
            if len(dyn_fields) == 0:
                raise ValueError("PinocchioStateBuilder: dynamics_fields must be non-empty when provided.")
            for field in dyn_fields:
                if field == "torque":
                    self.register_value_and_jac(
                        dtype=DTYPE_DYNAMICS,
                        owner_type=self.dynamics_owner_type,
                        field="torque",
                        value_handler=self._handle_torque,
                        jac_handler=self._handle_torque_jac,
                        jacobian_wrt=STATE_JACOBIAN_VAR,
                    )
                    continue
                if field == "momentum":
                    self.register_value_and_jac(
                        dtype=DTYPE_DYNAMICS,
                        owner_type=self.dynamics_owner_type,
                        field="momentum",
                        value_handler=self._handle_momentum,
                        jac_handler=self._handle_momentum_jac,
                        jacobian_wrt=STATE_JACOBIAN_VAR,
                    )
                    continue
                if field == "force":
                    # Joint-space generalized force alias. In this backend it maps to torque.
                    self.register_value_and_jac(
                        dtype=DTYPE_DYNAMICS,
                        owner_type=self.dynamics_owner_type,
                        field="force",
                        value_handler=self._handle_force,
                        jac_handler=self._handle_force_jac,
                        jacobian_wrt=STATE_JACOBIAN_VAR,
                    )
                    continue
                if dynamics_custom_handlers is not None and field in dynamics_custom_handlers:
                    value_handler, jac_handler = dynamics_custom_handlers[field]
                    self.register_value_and_jac(
                        dtype=DTYPE_DYNAMICS,
                        owner_type=self.dynamics_owner_type,
                        field=field,
                        value_handler=value_handler,
                        jac_handler=jac_handler,
                        jacobian_wrt=STATE_JACOBIAN_VAR,
                    )
                    continue
                raise ValueError(
                    f"PinocchioStateBuilder: unsupported dynamics field {field!r}. "
                    "Currently supported: 'torque', 'momentum', 'force' "
                    "(plus dynamics_custom_handlers)."
                )

    def _update_kinematics(self, q: Array) -> None:
        pin.forwardKinematics(self.model, self.data, q)
        if hasattr(pin, "computeJointJacobians"):
            pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def _resolve_state_ref(self, key: StateKey) -> Any:
        owner = getattr(key, "owner", None)
        owner_type = getattr(owner, "owner_type", None)
        owner_name = getattr(owner, "owner_name", None)
        if not isinstance(owner_name, str) or owner_name == "":
            raise ValueError(f"Pinocchio backend expects non-empty owner_name in key, got: {key!r}")

        if owner_type == self.owner_type:
            return int(self.model.getFrameId(owner_name))

        if owner_type == self.dynamics_owner_type and getattr(key, "dtype", None) == DTYPE_DYNAMICS:
            return ("dynamics", owner_name)

        if (
            owner_type == "total_joint"
            and getattr(key, "dtype", None) == DTYPE_COORD
            and str(getattr(key, "field", "")) == "q"
        ):
            return ("total_joint", owner_name, "q")

        raise ValueError(
            "Pinocchio backend expects owner_type="
            f"{self.owner_type!r} (kinematics) or {self.dynamics_owner_type!r} (dynamics), got: {key!r}"
        )

    def _frame_pos(self, frame_ref: Any) -> Array:
        frame_id = int(frame_ref)
        return np.asarray(self.data.oMf[frame_id].translation, dtype=float).reshape(3)

    def _frame_rot(self, frame_ref: Any) -> Array:
        frame_id = int(frame_ref)
        rot = np.asarray(self.data.oMf[frame_id].rotation, dtype=float).reshape(3, 3)
        return rot.reshape(-1)

    def _frame_value(self, field: str, frame_ref: Any) -> Array:
        if field == "pos":
            return self._frame_pos(frame_ref)
        if field == "rot":
            return self._frame_rot(frame_ref)
        if field == "frame":
            return np.concatenate([self._frame_pos(frame_ref), self._frame_rot(frame_ref)], axis=0)
        raise ValueError(f"PinocchioStateBuilder: unsupported value field: {field!r}")

    def _finite_difference_jacobian(
        self,
        q: Array,
        *,
        frame_ref: Any,
        value_fn: Callable[[Any], Array],
    ) -> Array:
        q0 = np.asarray(q, dtype=float).reshape(-1)
        y0 = np.asarray(value_fn(frame_ref), dtype=float).reshape(-1)
        m = int(y0.size)
        n = int(q0.size)
        J = np.zeros((m, n), dtype=float)
        if n == 0:
            return J

        eps = float(self.finite_diff_eps)
        try:
            for i in range(n):
                h = eps * max(1.0, abs(float(q0[i])))
                q_plus = q0.copy()
                q_plus[i] += h
                self._update_kinematics(q_plus)
                y_plus = np.asarray(value_fn(frame_ref), dtype=float).reshape(-1)
                J[:, i] = (y_plus - y0) / h
        finally:
            self._update_kinematics(q0)

        return J

    def _handle_pos(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del q, key
        return self._frame_pos(frame_ref)

    def _handle_pos_jac(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del key
        J6 = compute_pinocchio_frame_jacobian(self.model, self.data, q, frame_ref)
        return linear_part_from_jacobian6(J6, order=self.jac6_order)

    def _handle_rot(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del q, key
        return self._frame_rot(frame_ref)

    def _handle_rot_jac(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del key
        return self._finite_difference_jacobian(
            q,
            frame_ref=frame_ref,
            value_fn=self._frame_rot,
        )

    def _handle_frame(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del q, key
        return self._frame_value("frame", frame_ref)

    def _handle_frame_jac(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del key
        return self._finite_difference_jacobian(
            q,
            frame_ref=frame_ref,
            value_fn=lambda ref: self._frame_value("frame", ref),
        )

    def _torque_value(self, q: Array) -> Array:
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        if hasattr(pin, "computeGeneralizedGravity"):
            tau = pin.computeGeneralizedGravity(self.model, self.data, q_vec)
            return _as_dyn_vec(tau)
        if hasattr(pin, "rnea"):
            nv = int(getattr(self.model, "nv", q_vec.size))
            v = np.zeros((nv,), dtype=float)
            a = np.zeros((nv,), dtype=float)
            tau = pin.rnea(self.model, self.data, q_vec, v, a)
            return _as_dyn_vec(tau)
        raise ValueError("PinocchioStateBuilder: torque computation requires computeGeneralizedGravity or rnea.")

    def _handle_torque(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        return self._torque_value(q)

    def _handle_torque_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key
        q0 = np.asarray(q, dtype=float).reshape(-1)
        y0 = self._torque_value(q0)
        m = int(y0.size)
        n = int(q0.size)
        J = np.zeros((m, n), dtype=float)
        if n == 0:
            return J

        eps = float(self.finite_diff_eps)
        for i in range(n):
            h = eps * max(1.0, abs(float(q0[i])))
            qp = q0.copy()
            qp[i] += h
            yp = self._torque_value(qp)
            J[:, i] = (yp - y0) / h
        return J

    def _momentum_value(self, q: Array) -> Array:
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        nv = int(getattr(self.model, "nv", q_vec.size))
        v0 = np.zeros((nv,), dtype=float)

        if hasattr(pin, "computeCentroidalMomentum"):
            h = pin.computeCentroidalMomentum(self.model, self.data, q_vec, v0)
            return _as_dyn_vec(h)

        if hasattr(pin, "ccrba"):
            h = pin.ccrba(self.model, self.data, q_vec, v0)
            if h is None:
                h = getattr(self.data, "hg", None)
            if h is None:
                raise ValueError("PinocchioStateBuilder: ccrba did not provide centroidal momentum.")
            return _as_dyn_vec(h)

        raise ValueError("PinocchioStateBuilder: momentum computation requires computeCentroidalMomentum or ccrba.")

    def _handle_momentum(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        return self._momentum_value(q)

    def _handle_momentum_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        q0 = np.asarray(q, dtype=float).reshape(-1)
        y0 = self._momentum_value(q0)
        m = int(y0.size)
        n = int(q0.size)
        J = np.zeros((m, n), dtype=float)
        if n == 0:
            return J

        eps = float(self.finite_diff_eps)
        for i in range(n):
            h = eps * max(1.0, abs(float(q0[i])))
            qp = q0.copy()
            qp[i] += h
            yp = self._momentum_value(qp)
            J[:, i] = (yp - y0) / h
        return J

    def _handle_force(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        return self._torque_value(q)

    def _handle_force_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        return self._handle_torque_jac(q, key, state_ref)


class PinocchioTrajectoryStateBuilder(PinocchioStateBuilder):
    """Pinocchio trajectory builder with trajectory parameterization.

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
        jac6_order: Jacobian6Order = "linear_angular",
        finite_diff_eps: float = 1e-8,
        fields: Sequence[str] | None = None,
        dynamics_fields: Sequence[str] | None = None,
        dynamics_owner_type: str = "total_joint",
        dynamics_custom_handlers: Mapping[str, tuple[Callable[..., Array], Callable[..., Array]]] | None = None,
    ) -> None:
        self.trajectory_map = trajectory_map
        self.trajectory_derivative_maps: dict[int, TrajectoryMap] = {0: trajectory_map}
        if trajectory_derivative_maps is not None:
            for order_raw, traj in trajectory_derivative_maps.items():
                order = int(order_raw)
                if order < 0:
                    raise ValueError(
                        f"PinocchioTrajectoryStateBuilder: derivative order must be >= 0, got {order}."
                    )
                self.trajectory_derivative_maps[order] = traj
        self._validate_derivative_maps()

        super().__init__(
            model,
            data,
            q_var=p_var,
            jac6_order=jac6_order,
            finite_diff_eps=finite_diff_eps,
            fields=fields,
            dynamics_fields=dynamics_fields,
            dynamics_owner_type=dynamics_owner_type,
            dynamics_custom_handlers=dynamics_custom_handlers,
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
            raise ValueError("PinocchioTrajectoryStateBuilder: derivative map for order 0 is required.")
        for order, traj in self.trajectory_derivative_maps.items():
            if traj.p_dim != base.p_dim:
                raise ValueError(
                    "PinocchioTrajectoryStateBuilder: derivative map p_dim mismatch. "
                    f"order={order}, expected {base.p_dim}, got {traj.p_dim}."
                )
            if traj.steps != base.steps or traj.q_dim != base.q_dim:
                raise ValueError(
                    "PinocchioTrajectoryStateBuilder: derivative map shape mismatch. "
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

    def _chain_param_jac(
        self,
        J_raw: Array,
        *,
        key: StateKey,
        jacobian_wrt: str | None,
        dqdp_k: Array,
    ) -> Array:
        Jm = np.asarray(J_raw, dtype=float)
        if Jm.ndim != 2:
            raise ValueError(
                f"PinocchioTrajectoryStateBuilder: Jacobian must be 2D, got shape {Jm.shape} for key {key!r}."
            )

        wrt = None if jacobian_wrt is None else str(jacobian_wrt)
        if wrt == self.q_var:
            return Jm

        if wrt == STATE_JACOBIAN_VAR:
            if Jm.shape[1] == dqdp_k.shape[0]:
                return Jm @ dqdp_k
            raise ValueError(
                "PinocchioTrajectoryStateBuilder: Jacobian chain mismatch. "
                f"J_raw has shape {Jm.shape}, dqdp has shape {dqdp_k.shape}."
            )

        raise ValueError(
            "PinocchioTrajectoryStateBuilder: unsupported jacobian_wrt metadata for parameter chain. "
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
                "PinocchioTrajectoryStateBuilder: time grid mismatch. "
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
                "PinocchioTrajectoryStateBuilder: parameter size mismatch. "
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
            q_k = np.asarray(self.trajectory_map.q_at(p, k), dtype=float).reshape(-1)
            dqdp_k = np.asarray(self.trajectory_map.dqdp_at(k), dtype=float)
            self._update_kinematics(q_k)

            for key, entry in grouped[k]:
                state_ref = self._state_ref(key, state_ref_field=entry.state_ref_field)
                value = entry.handler(q_k, key, state_ref)
                if self._is_param_jac_key(key):
                    value = self._chain_param_jac(
                        value,
                        key=key,
                        jacobian_wrt=entry.jacobian_wrt,
                        dqdp_k=dqdp_k,
                    )
                out[key] = value

        return out


def _infer_model_dof(model: Any) -> int | None:
    dof_fn = getattr(model, "dof", None)
    if callable(dof_fn):
        try:
            return int(dof_fn())
        except Exception:
            return None
    for attr in ("nq", "nv"):
        raw = getattr(model, attr, None)
        if raw is None:
            continue
        try:
            return int(raw)
        except Exception:
            continue
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
        raise ValueError("compile_pinocchio_trajectory_problem: dynamics_fields must be non-empty when provided.")
    return tuple(out)


def _registered_dynamics_base_fields(
    *,
    builder: PinocchioTrajectoryStateBuilder,
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


def _validate_pinocchio_runtime_dynamics_coverage(
    *,
    runtime: ProblemRuntime,
    builder: PinocchioTrajectoryStateBuilder,
    dynamics_owner_type: str,
) -> None:
    requested_fields, unsupported_owner_types = _required_dynamics_base_fields(
        runtime=runtime,
        owner_type=dynamics_owner_type,
    )
    if unsupported_owner_types:
        unsupported = ", ".join(sorted(unsupported_owner_types))
        raise ValueError(
            "compile_pinocchio_trajectory_problem: DSL contains dynamics keys with unsupported owner_type(s): "
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
        "compile_pinocchio_trajectory_problem: DSL requests dynamics field(s) that are not registered in "
        "PinocchioTrajectoryStateBuilder. "
        f"Missing: {missing_str}. Requested: {requested_str}. Registered: {registered_str}. "
        "Add missing entries to `dynamics_fields` (or dynamics_custom_handlers), "
        "or remove corresponding get_state dynamics terms."
    )


@dataclass
class _PinocchioTrajectoryCompileAdapter:
    jac6_order: Jacobian6Order = "linear_angular"
    finite_diff_eps: float = 1e-8
    fields: Sequence[str] | None = None
    dynamics_fields: Sequence[str] | None = None
    dynamics_owner_type: str = "total_joint"
    dynamics_custom_handlers: Mapping[str, tuple[Callable[..., Array], Callable[..., Array]]] | None = None
    resolved_dynamics_fields: tuple[str, ...] = ()

    def infer_model_dof(self, model: Any) -> int | None:
        return _infer_model_dof(model)

    def infer_model_order(self, model: Any) -> int:
        return _infer_model_order(model)

    def _resolve_dynamics_fields(
        self,
        *,
        model: Any,
        data: Any,
        prepared: PreparedTrajectoryProblemDsl,
    ) -> tuple[str, ...] | None:
        dynamics_fields_use = _canonicalize_dynamics_fields(self.dynamics_fields)
        if dynamics_fields_use is None:
            probe_builder = PinocchioTrajectoryStateBuilder(
                model,
                data,
                trajectory_map=prepared.trajectory_map,
                trajectory_derivative_maps=prepared.trajectory_derivative_maps,
                p_var=prepared.p_var,
                jac6_order=self.jac6_order,
                finite_diff_eps=self.finite_diff_eps,
                fields=self.fields,
                dynamics_fields=None,
                dynamics_owner_type=self.dynamics_owner_type,
                dynamics_custom_handlers=self.dynamics_custom_handlers,
            )
            probe_runtime = compile_problem(prepared.dsl, build_state=probe_builder.build_state)
            requested_fields_order, unsupported_owner_types = _required_dynamics_base_fields_in_order(
                runtime=probe_runtime,
                owner_type=self.dynamics_owner_type,
            )
            if unsupported_owner_types:
                unsupported = ", ".join(sorted(unsupported_owner_types))
                raise ValueError(
                    "compile_pinocchio_trajectory_problem: DSL contains dynamics keys with unsupported owner_type(s): "
                    f"{unsupported}. Supported owner_type is {self.dynamics_owner_type!r}."
                )
            dynamics_fields_use = tuple(requested_fields_order) if len(requested_fields_order) > 0 else None
        return dynamics_fields_use

    def build_state_builder(
        self,
        *,
        model: Any,
        data: Any,
        prepared: PreparedTrajectoryProblemDsl,
    ) -> PinocchioTrajectoryStateBuilder:
        dynamics_fields_use = self._resolve_dynamics_fields(
            model=model,
            data=data,
            prepared=prepared,
        )
        self.resolved_dynamics_fields = (
            tuple() if dynamics_fields_use is None else tuple(dynamics_fields_use)
        )
        return PinocchioTrajectoryStateBuilder(
            model,
            data,
            trajectory_map=prepared.trajectory_map,
            trajectory_derivative_maps=prepared.trajectory_derivative_maps,
            p_var=prepared.p_var,
            jac6_order=self.jac6_order,
            finite_diff_eps=self.finite_diff_eps,
            fields=self.fields,
            dynamics_fields=dynamics_fields_use,
            dynamics_owner_type=self.dynamics_owner_type,
            dynamics_custom_handlers=self.dynamics_custom_handlers,
        )

    def validate_runtime(
        self,
        *,
        runtime: ProblemRuntime,
        state_builder: PinocchioTrajectoryStateBuilder,
        prepared: PreparedTrajectoryProblemDsl,
    ) -> None:
        del prepared
        _validate_pinocchio_runtime_dynamics_coverage(
            runtime=runtime,
            builder=state_builder,
            dynamics_owner_type=self.dynamics_owner_type,
        )


def compile_pinocchio_trajectory_problem(
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
    jac6_order: Jacobian6Order = "linear_angular",
    finite_diff_eps: float = 1e-8,
    fields: Sequence[str] | None = None,
    dynamics_fields: Sequence[str] | None = None,
    dynamics_owner_type: str = "total_joint",
    dynamics_custom_handlers: Mapping[str, tuple[Callable[..., Array], Callable[..., Array]]] | None = None,
) -> PinocchioTrajectoryCompiledProblem:
    adapter = _PinocchioTrajectoryCompileAdapter(
        jac6_order=jac6_order,
        finite_diff_eps=finite_diff_eps,
        fields=fields,
        dynamics_fields=dynamics_fields,
        dynamics_owner_type=dynamics_owner_type,
        dynamics_custom_handlers=dynamics_custom_handlers,
    )
    compiled = compile_trajectory_problem_with_adapter(
        dsl,
        model=model,
        data=data,
        adapter=adapter,
        p_var=p_var,
        max_derivative_order=max_derivative_order,
        derivative_wrt=derivative_wrt,
        default_steps=default_steps,
        default_q_dim=default_q_dim,
        default_dt=default_dt,
    )
    return PinocchioTrajectoryCompiledProblem(
        runtime=compiled.runtime,
        trajectory_map=compiled.prepared.trajectory_map,
        trajectory_derivative_maps=compiled.prepared.trajectory_derivative_maps,
        p_var=compiled.prepared.p_var,
        dt=float(compiled.prepared.dt),
        model_order=int(compiled.prepared.model_order),
        dynamics_fields=tuple(adapter.resolved_dynamics_fields),
    )


def _as_dyn_vec(x: Any) -> Array:
    if hasattr(x, "vector"):
        return np.asarray(getattr(x, "vector"), dtype=float).reshape(-1)
    return np.asarray(x, dtype=float).reshape(-1)
