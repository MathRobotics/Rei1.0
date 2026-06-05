from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    torque_derivative_order,
)
from ....core.trajectory import TrajectoryMap
from ..dispatch.template import BackendDispatchStateBuilder
from .motion import (
    interleaved_motion_jacobian_used_order,
)
from ..trajectory import (
    chain_param_jacobian,
    compose_interleaved_motion_and_jac,
    TrajectoryStateBuilderMixin,
    unique_jacobian_chain_candidates,
    validate_trajectory_derivative_maps,
)
from . import kots_api as kapi
from .kots_api import StateType
from .kots_adapter import KotsAdapter, TotalJointDynamicsStateRef
from .provider import register_robot_binding_table

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


# kots.py 内で「どの field ファミリを提供するか」を宣言する登録表。
KOTS_DEFAULT_BINDINGS: dict[str, str] = {
    "kinematics.link.pos": "value",
    "kinematics.link.pos.J_state": "jac",
    "kinematics.link.rot": "value",
    "kinematics.link.rot.J_state": "jac",
    "kinematics.link.frame": "value",
    "kinematics.link.frame.J_state": "jac",
}


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
        self.prefer_matvec_jacobian = bool(prefer_matvec_jacobian)
        self._jacobian_ops = kapi.RoboKotsJacobianOperator(self.model)
        self.adapter = KotsAdapter(self, state_type=StateType)

        supported_fields = {
            key.split(".")[2]
            for key in KOTS_DEFAULT_BINDINGS
            if not key.endswith(".J_state")
        }
        selected_fields = sorted(supported_fields) if fields is None else [str(f) for f in fields]
        if len(selected_fields) == 0:
            raise ValueError("KotsStateBuilder: fields must be non-empty.")

        self.field_to_jac: dict[str, str] = {}
        kinematics_bindings: dict[str, str] = {}
        for field_raw in selected_fields:
            field = canonical_field_name(field_raw)
            if field not in supported_fields:
                supported = ", ".join(sorted(supported_fields))
                raise ValueError(
                    f"KotsStateBuilder: unsupported field {field!r}. "
                    f"Supported fields: {supported}."
                )
            kinematics_bindings[f"kinematics.link.{field}"] = KOTS_DEFAULT_BINDINGS[f"kinematics.link.{field}"]
            kinematics_bindings[f"kinematics.link.{field}.J_state"] = KOTS_DEFAULT_BINDINGS[
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
                register_robot_binding_table(
                    self,
                    {
                        f"dynamics.{self.dynamics_owner_type}.{field}": "value",
                        f"dynamics.{self.dynamics_owner_type}.{field}.J_state": "jac",
                    },
                    handler_owner=self.adapter,
                    owner_types=("total_joint", self.dynamics_owner_type),
                    default_jacobian_wrt=STATE_JACOBIAN_VAR,
                )

    def _update_dynamics_if_available(self) -> bool:
        return self.adapter.update_dynamics_if_available()

    def _update_kinematics(self, q: Array) -> None:
        self.adapter.update_kinematics(q)

    def _model_dof(self) -> int:
        return self.adapter.model_dof()

    def _model_order(self) -> int:
        return self.adapter.model_order()

    def _expand_coordinate_motion(self, q: Array, *, dof: int, order: int) -> Array:
        return self.adapter.expand_coordinate_motion(q, dof=dof, order=order)

    def _resolve_state_ref(self, key: StateKey) -> Any:
        owner = getattr(key, "owner", None)
        owner_type = getattr(owner, "owner_type", None)
        owner_name = getattr(owner, "owner_name", None)
        if not isinstance(owner_name, str) or owner_name == "":
            raise ValueError(f"Kots backend expects non-empty owner_name in key, got: {key!r}")

        state_field = self.adapter.state_field_name(key.field)
        if owner_type == "total_joint" and getattr(key, "dtype", None) == DTYPE_COORD and key.field == "q":
            # Joint-q terms are computed directly from optimization variables; no backend state query is required.
            return ("total_joint", owner_name, "q")

        if owner_type == self.dynamics_owner_type and getattr(key, "dtype", None) == DTYPE_DYNAMICS:
            # RoboKots does not robustly support world dynamics queries for owner_type="total_joint".
            # Expand to per-joint queries and stack them in dof order.
            joint_refs = self.adapter.resolve_total_joint_dynamics_refs(state_field=state_field, key=key)
            if joint_refs is not None:
                return TotalJointDynamicsStateRef(field=state_field, refs=joint_refs)

        route = self._route_for_key(key)
        if route is None or route not in self._dispatch:
            raise ValueError(f"Kots backend has no handler route for key: {key!r}")

        frame_name = getattr(key, "frame", None) or "world"
        return self.adapter.make_state_type(
            owner_type=str(owner_type),
            owner_name=owner_name,
            state_field=state_field,
            frame_name=str(frame_name),
        )

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
        rhs_local = self.adapter.rotate_link_kinematics_rhs_to_local(
            rhs=np.asarray(rhs, dtype=float),
            key=key,
            state_ref=state_ref,
        )
        return self.adapter.transpose_matvec_from_state_ref(state_ref, rhs_local)


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
            return self.adapter.jac(np.zeros((dqdp_k.shape[0],), dtype=float), key, state_ref)
        if wrt != STATE_JACOBIAN_VAR:
            raise ValueError(
                "KotsTrajectoryStateBuilder: unsupported jacobian_wrt metadata for parameter chain. "
                f"Expected {self.q_var!r} or {STATE_JACOBIAN_VAR!r}, got {wrt!r}."
            )

        total_joint_ref = self.adapter.as_total_joint_dynamics_state_ref(state_ref)
        last_error: Exception | None = None
        for cols in self._motion_jacobian_chain_candidates(dqdp_k=dqdp_k, dmotiondp_k=dmotiondp_k, key=key):
            try:
                if total_joint_ref is None:
                    Jp = self.adapter.jac_from_matvec_single_state_ref(state_ref, cols)
                    return self.adapter.rotate_link_kinematics_jacobian_to_world(J=Jp, key=key, state_ref=state_ref)

                try:
                    return self.adapter.jac_from_matvec_single_state_ref(list(total_joint_ref.refs), cols)
                except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
                    pass

                blocks = [self.adapter.jac_from_matvec_single_state_ref(ref, cols) for ref in total_joint_ref.refs]
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
        rhs_local = self.adapter.rotate_link_kinematics_rhs_to_local(
            rhs=np.asarray(rhs, dtype=float),
            key=key,
            state_ref=state_ref,
        )
        motion_grad = np.asarray(self.adapter.transpose_matvec_from_state_ref(state_ref, rhs_local), dtype=float)

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
    "KOTS_DEFAULT_BINDINGS",
    "KotsStateBuilder",
    "KotsTrajectoryStateBuilder",
]
