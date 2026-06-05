from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ....core.state_cache import StateKey
from ....core.state_schema import (
    DTYPE_COORD,
    DTYPE_KINEMATICS,
    canonical_field_name,
)
from ....core.trajectory import TrajectoryMap
from .spatial import Jacobian6Order
from ..trajectory import (
    chain_param_jacobian,
    compose_stacked_motion_and_jac,
    TrajectoryStateBuilderMixin,
    validate_trajectory_derivative_maps,
)
from ..dispatch.template import BackendDispatchStateBuilder
from .pinocchio_adapter import compute_pinocchio_frame_jacobian, PinocchioAdapter
from .provider import register_robot_binding_table

Array = np.ndarray
STATE_JACOBIAN_VAR = "state"


# pinocchio.py 内で「どの field ファミリを提供するか」を宣言する登録表。
# StateKey の (dtype, owner_type, field) マッチは BackendDispatchStateBuilder 側で自動実行される。
PINOCCHIO_DEFAULT_BINDINGS: dict[str, str] = {
    "kinematics.link.pos": "pos",
    "kinematics.link.pos.J_state": "pos_jac",
    "kinematics.link.rot": "rot",
    "kinematics.link.rot.J_state": "rot_jac",
    "kinematics.link.frame": "frame",
    "kinematics.link.frame.J_state": "frame_jac",
}

PINOCCHIO_DYNAMICS_BINDINGS: dict[str, tuple[str, str]] = {
    "torque": ("torque", "torque_jac"),
    "momentum": ("momentum", "momentum_jac"),
    "force": ("force", "force_jac"),
}


class PinocchioStateBuilder(BackendDispatchStateBuilder):
    """Pinocchio -> `build_state()` bridge with StateKey-based automatic dispatch.

    `PINOCCHIO_DEFAULT_BINDINGS` をもとに、
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
        torque_jacobian: str = "auto",
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
        torque_jacobian_name = str(torque_jacobian).strip().lower()
        if torque_jacobian_name not in ("auto", "analytic", "finite_difference"):
            raise ValueError(
                "PinocchioStateBuilder: torque_jacobian must be one of "
                "'auto', 'analytic', or 'finite_difference'."
            )
        self.torque_jacobian = torque_jacobian_name
        self.adapter = PinocchioAdapter(
            self.model,
            self.data,
            owner_type=self.owner_type,
            dynamics_owner_type=self.dynamics_owner_type,
            jac6_order=self.jac6_order,
            finite_diff_eps=self.finite_diff_eps,
            torque_jacobian=self.torque_jacobian,
        )

        supported_fields = {
            key.split(".")[2]
            for key in PINOCCHIO_DEFAULT_BINDINGS
            if not key.endswith(".J_state")
        }
        selected_fields = sorted(supported_fields) if fields is None else [str(f) for f in fields]
        if len(selected_fields) == 0:
            raise ValueError("PinocchioStateBuilder: fields must be non-empty.")

        self.field_to_jac: dict[str, str] = {}
        kinematics_bindings: dict[str, str] = {}
        for field_raw in selected_fields:
            field = canonical_field_name(field_raw)
            if field not in supported_fields:
                supported = ", ".join(sorted(supported_fields))
                raise ValueError(
                    f"PinocchioStateBuilder: unsupported field {field!r}. "
                    f"Supported fields: {supported}."
                )
            kinematics_bindings[f"kinematics.link.{field}"] = PINOCCHIO_DEFAULT_BINDINGS[f"kinematics.link.{field}"]
            kinematics_bindings[f"kinematics.link.{field}.J_state"] = PINOCCHIO_DEFAULT_BINDINGS[
                f"kinematics.link.{field}.J_state"
            ]
        registered = register_robot_binding_table(
            self,
            kinematics_bindings,
            handler_owner=self.adapter,
            default_jacobian_wrt=STATE_JACOBIAN_VAR,
        )
        for (_dtype, _owner_type, field), jac_name in registered.items():
            if jac_name is not None:
                self.field_to_jac[field] = jac_name

        if dynamics_fields is not None:
            dyn_fields = [canonical_field_name(str(f)) for f in dynamics_fields]
            dyn_fields = list(dict.fromkeys(dyn_fields))
            if len(dyn_fields) == 0:
                raise ValueError("PinocchioStateBuilder: dynamics_fields must be non-empty when provided.")
            for field in dyn_fields:
                if field in PINOCCHIO_DYNAMICS_BINDINGS:
                    value_name, jac_name = PINOCCHIO_DYNAMICS_BINDINGS[field]
                    register_robot_binding_table(
                        self,
                        {
                            f"dynamics.{self.dynamics_owner_type}.{field}": value_name,
                            f"dynamics.{self.dynamics_owner_type}.{field}.J_state": jac_name,
                        },
                        handler_owner=self.adapter,
                        owner_types=("total_joint", self.dynamics_owner_type),
                        default_jacobian_wrt=STATE_JACOBIAN_VAR,
                    )
                    continue
                if dynamics_custom_handlers is not None and field in dynamics_custom_handlers:
                    value_handler, jac_handler = dynamics_custom_handlers[field]
                    register_robot_binding_table(
                        self,
                        {
                            f"dynamics.{self.dynamics_owner_type}.{field}": value_handler,
                            f"dynamics.{self.dynamics_owner_type}.{field}.J_state": jac_handler,
                        },
                        owner_types=("total_joint", self.dynamics_owner_type),
                        default_jacobian_wrt=STATE_JACOBIAN_VAR,
                    )
                    continue
                raise ValueError(
                    f"PinocchioStateBuilder: unsupported dynamics field {field!r}. "
                    "Currently supported: 'torque', 'momentum', 'force' "
                    "(plus dynamics_custom_handlers)."
                )

    def _update_kinematics(self, q: Array) -> None:
        self.adapter.update(q)

    def _resolve_state_ref(self, key: StateKey) -> Any:
        return self.adapter.resolve_ref(key)

class PinocchioTrajectoryStateBuilder(TrajectoryStateBuilderMixin, PinocchioStateBuilder):
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
        torque_jacobian: str = "auto",
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
        self._active_step_k: int | None = None
        self._active_motion: Array | None = None

        super().__init__(
            model,
            data,
            q_var=p_var,
            jac6_order=jac6_order,
            finite_diff_eps=finite_diff_eps,
            torque_jacobian=torque_jacobian,
            fields=fields,
            dynamics_fields=dynamics_fields,
            dynamics_owner_type=dynamics_owner_type,
            dynamics_custom_handlers=dynamics_custom_handlers,
        )
        self.adapter.motion_provider = lambda q, key: self._active_motion_triplet(q=q, key=key)
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
            error_prefix="PinocchioTrajectoryStateBuilder",
        )

    def _handle_joint_q_value(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        return np.asarray(q, dtype=float).reshape(-1).copy()

    def _handle_joint_q_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        n = int(np.asarray(q, dtype=float).reshape(-1).size)
        return np.eye(n, dtype=float)

    def _compose_motion_and_jac(self, p: Array, *, k: int) -> tuple[Array, Array]:
        return compose_stacked_motion_and_jac(
            p,
            trajectory_map=self.trajectory_map,
            trajectory_derivative_maps=self.trajectory_derivative_maps,
            derivative_orders=(0, 1, 2),
            k=k,
            error_prefix="PinocchioTrajectoryStateBuilder",
        )

    def _active_motion_triplet(self, *, q: Array, key: StateKey) -> tuple[Array, Array, Array]:
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        dof = int(q_vec.size)

        if self._active_step_k is None or self._active_motion is None:
            nv = self.adapter.pin_nv(default=dof)
            return q_vec, np.zeros((nv,), dtype=float), np.zeros((nv,), dtype=float)

        if int(self._active_step_k) != int(getattr(key, "k", -1)):
            raise RuntimeError(
                "PinocchioTrajectoryStateBuilder: active step mismatch while evaluating dynamics. "
                f"active_k={self._active_step_k}, key.k={getattr(key, 'k', None)}."
            )

        motion = np.asarray(self._active_motion, dtype=float).reshape(-1)
        expected = int(3 * dof)
        if motion.size != expected:
            raise ValueError(
                "PinocchioTrajectoryStateBuilder: active motion size mismatch. "
                f"Expected {expected}, got {motion.size}."
            )

        q_use = motion[0:dof].copy()
        dq_use = motion[dof : 2 * dof].copy()
        ddq_use = motion[2 * dof : 3 * dof].copy()
        return q_use, dq_use, ddq_use

    def _update_trajectory_step(self, *, k: int, q_k: Array, motion_k: Array) -> None:
        self._active_step_k = int(k)
        self._active_motion = np.asarray(motion_k, dtype=float).reshape(-1)
        self._update_kinematics(q_k)

    def _finalize_trajectory_build(self) -> None:
        self._active_step_k = None
        self._active_motion = None

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
                f"PinocchioTrajectoryStateBuilder: Jacobian must be 2D, got shape {Jm.shape} for key {key!r}."
            )

        wrt = None if jacobian_wrt is None else str(jacobian_wrt)
        if wrt == self.q_var:
            return Jm

        if wrt == STATE_JACOBIAN_VAR:
            return chain_param_jacobian(
                Jm,
                q_var=self.q_var,
                state_jacobian_var=STATE_JACOBIAN_VAR,
                key=key,
                jacobian_wrt=jacobian_wrt,
                dqdp=dqdp_k,
                dmotiondp=dmotiondp_k,
                error_prefix="PinocchioTrajectoryStateBuilder",
            )

        raise ValueError(
            "PinocchioTrajectoryStateBuilder: unsupported jacobian_wrt metadata for parameter chain. "
            f"Expected {self.q_var!r} or {STATE_JACOBIAN_VAR!r}, got {wrt!r}."
        )

__all__ = [
    "Jacobian6Order",
    "compute_pinocchio_frame_jacobian",
    "PinocchioAdapter",
    "PINOCCHIO_DEFAULT_BINDINGS",
    "PINOCCHIO_DYNAMICS_BINDINGS",
    "PinocchioStateBuilder",
    "PinocchioTrajectoryStateBuilder",
]
