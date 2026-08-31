from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
_KOTS_BACKENDS = ("numpy", "rust")
_KINETIC_ENERGY_FIELD = "kinetic_energy"
_KINETIC_ENERGY_OWNER_TYPE = "total_body"


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


def _normalize_kots_backend(backend: str | None) -> str | None:
    if backend is None:
        return None
    name = str(backend).strip().lower()
    if name == "":
        raise ValueError("KotsStateBuilder: kots_backend must be non-empty when provided.")
    if name not in _KOTS_BACKENDS:
        allowed = ", ".join(repr(v) for v in _KOTS_BACKENDS)
        raise ValueError(f"KotsStateBuilder: kots_backend must be one of {allowed}, got {backend!r}.")
    return name


def _normalize_kots_gravity(gravity: Sequence[float] | None) -> tuple[float, float, float] | None:
    if gravity is None:
        return None
    values = np.asarray(gravity, dtype=float)
    if values.shape != (3,):
        raise ValueError(f"KotsStateBuilder: gravity must have shape (3,), got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("KotsStateBuilder: gravity must contain only finite values.")
    return tuple(float(value) for value in values)


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
        kots_backend: str | None = None,
        gravity: Sequence[float] | None = None,
    ) -> None:
        super().__init__(model, data, q_var=q_var)
        self.dtype = DTYPE_KINEMATICS
        self.owner_type = "link"
        self.kots_backend = _normalize_kots_backend(kots_backend)
        self.gravity = _normalize_kots_gravity(gravity)
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
            for field in dyn_fields:
                if field == _KINETIC_ENERGY_FIELD:
                    # RoboKots exposes this scalar through the canonical
                    # StateType(total_body, total_body, kinetic_energy).
                    register_robot_binding_table(
                        self,
                        {
                            f"dynamics.{_KINETIC_ENERGY_OWNER_TYPE}.{field}": "value",
                            f"dynamics.{_KINETIC_ENERGY_OWNER_TYPE}.{field}.J_state": "jac",
                        },
                        handler_owner=self.adapter,
                        owner_types=(_KINETIC_ENERGY_OWNER_TYPE,),
                        default_jacobian_wrt=STATE_JACOBIAN_VAR,
                    )
                    continue
                self._needs_dynamics_update = True
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

        if (
            owner_type == _KINETIC_ENERGY_OWNER_TYPE
            and getattr(key, "dtype", None) == DTYPE_DYNAMICS
            and state_field == _KINETIC_ENERGY_FIELD
        ):
            # RoboKots recognizes whole-body kinetic energy only with the
            # frame-free canonical StateType(total_body, total_body, ...).
            return self.adapter.make_state_type(
                owner_type=str(owner_type),
                owner_name=owner_name,
                state_field=state_field,
                frame_name=None,
            )

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
        kots_backend: str | None = None,
        gravity: Sequence[float] | None = None,
        batch_trajectory: bool = True,
    ) -> None:
        self.trajectory_map = trajectory_map
        self.p_var = str(p_var)
        if self.p_var == "":
            raise ValueError("KotsTrajectoryStateBuilder: p_var must be non-empty.")
        self.jacobian_strategy = _normalize_kots_jacobian_strategy(
            jacobian_strategy,
            prefer_matvec_jacobian=prefer_matvec_jacobian,
        )
        self.batch_trajectory = bool(batch_trajectory)
        # The RoboKots model owns the materialized outward state.  Retain the
        # exact batch signature that was loaded so value/JVP/VJP phases of one
        # IOC evaluation can share it without another import+dynamics pass.
        self._batched_dynamics_cache_key: tuple[Any, ...] | None = None
        self._batched_dynamics_cache_hits = 0
        self._batched_dynamics_cache_misses = 0
        # None means unprobed.  Older RoboKots returns one fused VJP, whereas
        # IOC term scheduling requires one result per input request.
        self._batched_multi_vjp_contract: str | None = None
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
            kots_backend=kots_backend,
            gravity=gravity,
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

    def _compose_motion(self, p: Array, *, k: int) -> Array:
        p_vec = np.asarray(p, dtype=float).reshape(-1)
        dof = int(self.trajectory_map.q_dim)
        order = int(self._model_order())
        motion = np.zeros((dof * order,), dtype=float)
        for deriv_order, traj in self.trajectory_derivative_maps.items():
            deriv_order_i = int(deriv_order)
            if deriv_order_i >= order:
                continue
            q_r = np.asarray(traj.q_at(p_vec, k), dtype=float).reshape(-1)
            if q_r.size != dof:
                raise ValueError(
                    "KotsTrajectoryStateBuilder: derivative map q size mismatch. "
                    f"order={deriv_order_i}, expected {dof}, got {q_r.size}."
                )
            motion[deriv_order_i::order] = q_r
        return motion

    def _can_batch_trajectory_entries(self, grouped: Mapping[int, list[tuple[StateKey, Any]]]) -> bool:
        """Whether all requested trajectory entries can use RoboKots batch APIs.

        Coordinate entries are evaluated locally. Dynamics entries share one
        batched RoboKots state and are grouped by their StateType below.
        Kinematic entries retain the existing per-step path because world-frame
        rotation handling is currently state-local in Rei.
        """
        if not self.batch_trajectory or len(grouped) < 2:
            return False
        if not callable(getattr(self.model, "state_info_list", None)):
            return False
        if not callable(getattr(self.model, "jacobian_mul", None)):
            return False
        for entries in grouped.values():
            for key, _entry in entries:
                owner_type = getattr(getattr(key, "owner", None), "owner_type", None)
                if getattr(key, "dtype", None) == DTYPE_COORD and owner_type == "total_joint":
                    continue
                if getattr(key, "dtype", None) == DTYPE_DYNAMICS and owner_type in (
                    self.dynamics_owner_type,
                    _KINETIC_ENERGY_OWNER_TYPE,
                ):
                    continue
                return False
        return True

    def _batched_dynamics_key(self, *, p: Array, motions: Array, time: Any = None) -> tuple[Any, ...]:
        """Identity of the materialized RoboKots batch outward state.

        ``motions`` is included in addition to ``p`` because a caller may ask
        for a different subset/order of time steps at the same optimization
        point.  The explicit time, gravity, and model-order fields make cache
        invalidation independent of StateCache's revision bookkeeping.
        """
        p64 = np.ascontiguousarray(np.asarray(p, dtype=np.float64).reshape(-1))
        motions64 = np.ascontiguousarray(np.asarray(motions, dtype=np.float64))
        gravity = _normalize_kots_gravity(self.gravity)
        return (
            p64.shape,
            p64.tobytes(),
            motions64.shape,
            motions64.tobytes(),
            int(getattr(time, "revision", 0)) if time is not None else 0,
            int(getattr(time, "N", self.trajectory_map.steps - 1)) if time is not None else self.trajectory_map.steps - 1,
            float(getattr(time, "dt", 0.0)) if time is not None else None,
            gravity,
            int(self.adapter.model_order_cache_signature()),
        )

    def _update_batched_dynamics(self, motions: Array, *, p: Array, time: Any = None) -> None:
        key = self._batched_dynamics_key(p=p, motions=motions, time=time)
        if key == self._batched_dynamics_cache_key:
            self._batched_dynamics_cache_hits += 1
            return

        self.model.import_motions(np.asarray(motions, dtype=float))
        if self._needs_dynamics_update and not self.adapter.update_dynamics_if_available():
            raise AttributeError("RoboKots model does not expose a usable batched dynamics method.")
        self._batched_dynamics_cache_key = key
        self._batched_dynamics_cache_misses += 1

    def _batched_dynamics_value(self, state_ref: Any) -> Array:
        total_joint_ref = self.adapter.as_total_joint_dynamics_state_ref(state_ref)
        if total_joint_ref is not None:
            try:
                return np.asarray(self.model.state_info_list(list(total_joint_ref.refs)), dtype=float)
            except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
                pass
            parts = [np.asarray(self.model.state_info(ref), dtype=float) for ref in total_joint_ref.refs]
            return np.concatenate(parts, axis=-1)
        return np.asarray(self.model.state_info(state_ref), dtype=float)

    def _batched_dynamics_param_jacobian(
        self,
        *,
        key: StateKey,
        state_ref: Any,
        dmotiondps: Sequence[Array],
    ) -> Array:
        total_joint_ref = self.adapter.as_total_joint_dynamics_state_ref(state_ref)
        refs: Any = state_ref if total_joint_ref is None else list(total_joint_ref.refs)
        last_error: Exception | None = None
        for candidate_index in range(len(self._motion_jacobian_chain_candidates(
            dqdp_k=self.trajectory_map.dqdp_at(int(key.k)),
            dmotiondp_k=np.asarray(dmotiondps[0], dtype=float),
            key=key,
        ))):
            try:
                cols = np.stack(
                    [
                        self._motion_jacobian_chain_candidates(
                            dqdp_k=self.trajectory_map.dqdp_at(int(k)),
                            dmotiondp_k=np.asarray(dmotiondp, dtype=float),
                            key=key,
                        )[candidate_index]
                        for k, dmotiondp in zip(self._batch_ks, dmotiondps, strict=True)
                    ],
                    axis=0,
                )
                return np.asarray(self.model.jacobian_mul(refs, cols), dtype=float)
            except (AttributeError, KeyError, ValueError, TypeError, RuntimeError, IndexError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise AttributeError("RoboKots model does not expose a usable batched jacobian_mul method.")

    def _build_state_batched(
        self,
        *,
        p: Array,
        grouped: Mapping[int, list[tuple[StateKey, Any]]],
        time: Any = None,
    ) -> dict[StateKey, Any]:
        self._batch_ks = tuple(sorted(grouped))
        # A value-only state build must stay on the sparse trajectory path.
        # In particular, compose_interleaved_motion_and_jac calls dqdp_at(),
        # which is unnecessary for values and used to force all requested
        # time steps to construct parameter Jacobians.
        needs_param_jac = any(
            self._is_param_jac_key(key)
            for entries in grouped.values()
            for key, _entry in entries
        )
        if needs_param_jac:
            motions_and_jacs = [self._compose_motion_and_jac(p, k=k) for k in self._batch_ks]
            motions = np.stack([motion for motion, _jac in motions_and_jacs], axis=0)
            dmotiondps: list[Array] | None = [jac for _motion, jac in motions_and_jacs]
        else:
            motions = np.stack([self._compose_motion(p, k=k) for k in self._batch_ks], axis=0)
            dmotiondps = None
        q_values = [np.asarray(self.trajectory_map.q_at(p, k), dtype=float).reshape(-1) for k in self._batch_ks]
        self._update_batched_dynamics(motions, p=p, time=time)

        out: dict[StateKey, Any] = {}
        dynamic_groups: dict[tuple[Any, ...], list[tuple[int, StateKey, Any, Any]]] = {}
        for batch_index, k in enumerate(self._batch_ks):
            for key, entry in grouped[k]:
                state_ref = self._state_ref(key, state_ref_field=entry.state_ref_field)
                if getattr(key, "dtype", None) == DTYPE_COORD:
                    value = entry.handler(q_values[batch_index], key, state_ref)
                    if self._is_param_jac_key(key):
                        assert dmotiondps is not None
                        value = self._chain_param_jac(
                            value,
                            key=key,
                            jacobian_wrt=entry.jacobian_wrt,
                            dqdp_k=self.trajectory_map.dqdp_at(k),
                            dmotiondp_k=dmotiondps[batch_index],
                        )
                    out[key] = value
                    continue
                signature = (
                    entry.state_ref_field,
                    getattr(key, "dtype", None),
                    getattr(getattr(key, "owner", None), "owner_type", None),
                    getattr(getattr(key, "owner", None), "owner_name", None),
                    getattr(key, "frame", None),
                    self._is_param_jac_key(key),
                )
                dynamic_groups.setdefault(signature, []).append((batch_index, key, entry, state_ref))

        for entries in dynamic_groups.values():
            first_index, first_key, first_entry, first_ref = entries[0]
            if self._is_param_jac_key(first_key):
                if first_entry.jacobian_wrt != STATE_JACOBIAN_VAR:
                    raise ValueError("Batched RoboKots dynamics requires state-space Jacobian metadata.")
                assert dmotiondps is not None
                values = self._batched_dynamics_param_jacobian(
                    key=first_key,
                    state_ref=first_ref,
                    dmotiondps=dmotiondps,
                )
            else:
                values = self._batched_dynamics_value(first_ref)
            for batch_index, key, _entry, _state_ref in entries:
                out[key] = np.asarray(values[batch_index], dtype=float).copy()
        return out

    def build_state(
        self,
        x_all: Array,
        *,
        pack: Any = None,
        time: Any = None,
        required: Iterable[StateKey] | None = None,
    ) -> dict[StateKey, Any]:
        if required is None:
            return super().build_state(x_all, pack=pack, time=time, required=required)
        steps = self._expected_steps(time=time)
        grouped: dict[int, list[tuple[StateKey, Any]]] = {}
        for key in required:
            if not self._accept_required_key_for_traj(key, steps=steps):
                continue
            route = self._route_for_key(key)
            entry = None if route is None else self._dispatch.get(route)
            if entry is not None:
                grouped.setdefault(int(key.k), []).append((key, entry))
        if not self._can_batch_trajectory_entries(grouped):
            return super().build_state(x_all, pack=pack, time=time, required=required)
        p = self._extract_q(x_all, pack=pack)
        self._validate_trajectory_parameter_size(p)
        try:
            return self._build_state_batched(p=p, grouped=grouped, time=time)
        except (AttributeError, KeyError, ValueError, TypeError, RuntimeError, IndexError):
            # Preserve compatibility with older RoboKots versions and unusual
            # state combinations by falling back to the established path.
            return super().build_state(x_all, pack=pack, time=time, required=required)
        finally:
            self._batch_ks = ()

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
        deriv_order = torque_derivative_order(field)
        if deriv_order is None:
            return None
        return min(self._model_order(), deriv_order + 3)

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
    ) -> Array:
        rhs_local = self.adapter.rotate_link_kinematics_rhs_to_local(
            rhs=np.asarray(rhs, dtype=float),
            key=key,
            state_ref=state_ref,
        )
        motion_grad = np.asarray(self.adapter.transpose_matvec_from_state_ref(state_ref, rhs_local), dtype=float)
        mapped = self._trajectory_motion_gradient_transpose_at(k=int(key.k), motion_grad=motion_grad)
        if mapped is not None:
            return mapped
        raise ValueError(
            "KotsTrajectoryStateBuilder: transpose Jacobian chain mismatch. "
            f"transpose output has shape {motion_grad.shape}."
        )

    def _trajectory_motion_gradient_transpose_at(self, *, k: int, motion_grad: Array) -> Array | None:
        """Map an interleaved motion cotangent without building ``dmotiondp``."""
        grad = np.asarray(motion_grad, dtype=float)
        dof = self._model_dof()
        if dof <= 0 or grad.ndim not in (1, 2) or int(grad.shape[0]) % dof != 0:
            return None
        used_order = int(grad.shape[0] // dof)
        result_shape = (self.trajectory_map.p_dim,) if grad.ndim == 1 else (self.trajectory_map.p_dim, grad.shape[1])
        result = np.zeros(result_shape, dtype=float)
        for derivative_order in range(used_order):
            trajectory = self.trajectory_derivative_maps.get(derivative_order, None)
            if trajectory is not None:
                result += trajectory.apply_transpose_at(k, grad[derivative_order::used_order])
        return result

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
        motion_k = self._compose_motion(p, k=k)
        if update_kinematics:
            self._update_kinematics(motion_k)
        state_ref = self._state_ref(key, state_ref_field=entry.state_ref_field)
        return self._param_jac_transpose_mul_from_state_ref(
            key=key,
            state_ref=state_ref,
            rhs=rhs,
        )

    def param_jacobian_transpose_mul_many(
        self,
        x_all: Array,
        requests: Sequence[tuple[StateKey, Array]],
        *,
        pack: Any = None,
        time: Any = None,
    ) -> list[Array]:
        """Batched `J_p.T @ rhs` for compatible trajectory dynamics states.

        This is consumed by IOC's stacked state terms.  Unsupported state
        families deliberately use the existing single-state VJP path.
        """
        if len(requests) < 2 or not self.batch_trajectory:
            raise AttributeError("batched trajectory VJP requires at least two requests")
        if not callable(getattr(self.model, "jacobian_transpose_mul", None)):
            raise AttributeError("RoboKots model does not expose batched jacobian_transpose_mul")
        p = self._extract_q(np.asarray(x_all, dtype=float).reshape(-1), pack=pack)
        self._validate_trajectory_parameter_size(p)
        steps = self._expected_steps(time=time)

        prepared: list[tuple[StateKey, Array, Any, Any]] = []
        for key, rhs in requests:
            if not self._accept_required_key_for_traj(key, steps=steps):
                raise ValueError(f"KotsTrajectoryStateBuilder: invalid batched VJP key: {key!r}")
            if getattr(key, "dtype", None) != DTYPE_DYNAMICS:
                raise AttributeError("batched RoboKots VJP currently supports dynamics states only")
            route = self._route_for_key(key)
            entry = None if route is None else self._dispatch.get(route)
            if entry is None:
                raise AttributeError("batched RoboKots VJP requires a dynamics state handler")
            state_ref = self._state_ref(key, state_ref_field=entry.state_ref_field)
            prepared.append((key, np.asarray(rhs, dtype=float), entry, state_ref))

        groups: dict[tuple[Any, ...], list[tuple[int, StateKey, Array, Any]]] = {}
        for index, (key, rhs, entry, state_ref) in enumerate(prepared):
            signature = (
                entry.state_ref_field,
                getattr(getattr(key, "owner", None), "owner_type", None),
                getattr(getattr(key, "owner", None), "owner_name", None),
                getattr(key, "frame", None),
                rhs.shape[1:] if rhs.ndim >= 1 else (),
            )
            groups.setdefault(signature, []).append((index, key, rhs, state_ref))

        out: list[Array | None] = [None] * len(prepared)
        grouped_items = list(groups.values())

        # RoboKots >= the batched multi-VJP API can evaluate heterogeneous
        # state fields (for example torque and torque_d1) from the same outward
        # state.  The previous per-field loop issued one backend VJP per
        # state_ref_field even though the motion batch was identical.
        multi_vjp = getattr(self.model, "jacobian_transpose_mul_many", None)
        use_multi_vjp = (
            callable(multi_vjp)
            and len(grouped_items) >= 2
            and self._batched_multi_vjp_contract != "fused"
        )
        group_motions: list[list[Array]] = []
        if use_multi_vjp:
            for group in grouped_items:
                if len(group) < 2:
                    use_multi_vjp = False
                    break
                ks = [int(key.k) for _index, key, _rhs, _state_ref in group]
                group_motions.append([self._compose_motion(p, k=k) for k in ks])
            if use_multi_vjp:
                first_ks = [int(key.k) for _index, key, _rhs, _state_ref in grouped_items[0]]
                if any(
                    [int(key.k) for _index, key, _rhs, _state_ref in group] != first_ks
                    for group in grouped_items[1:]
                ):
                    use_multi_vjp = False

        if use_multi_vjp:
            self._update_batched_dynamics(
                np.stack(group_motions[0], axis=0),
                p=p,
                time=time,
            )
            backend_requests = []
            for group in grouped_items:
                first_ref = group[0][3]
                total_joint_ref = self.adapter.as_total_joint_dynamics_state_ref(first_ref)
                refs: Any = first_ref if total_joint_ref is None else list(total_joint_ref.refs)
                rhs_batch = np.stack([rhs for _index, _key, rhs, _state_ref in group], axis=0)
                backend_requests.append((refs, rhs_batch))
            try:
                raw_multi_vjp = multi_vjp(backend_requests)
                if isinstance(raw_multi_vjp, (list, tuple)):
                    grouped_motion_grads = list(raw_multi_vjp)
                elif (
                    isinstance(raw_multi_vjp, np.ndarray)
                    and raw_multi_vjp.ndim >= 1
                    and int(raw_multi_vjp.shape[0]) == len(grouped_items)
                ):
                    grouped_motion_grads = [raw_multi_vjp[index] for index in range(len(grouped_items))]
                else:
                    self._batched_multi_vjp_contract = "fused"
                    raise AttributeError("RoboKots multi-VJP returned one fused result")
                if len(grouped_motion_grads) != len(grouped_items):
                    self._batched_multi_vjp_contract = "fused"
                    raise ValueError("RoboKots multi-VJP must return one result for each input request.")
                for group, motions, motion_grads in zip(
                    grouped_items,
                    group_motions,
                    grouped_motion_grads,
                    strict=True,
                ):
                    self._chain_batched_param_vjp_group(
                        out=out,
                        group=group,
                        motions=motions,
                        motion_grads=motion_grads,
                    )
                self._batched_multi_vjp_contract = "per_input"
                return [np.asarray(value, dtype=float) for value in out]
            except (AttributeError, KeyError, ValueError, TypeError, RuntimeError, IndexError):
                # Older RoboKots releases return one fused VJP rather than one
                # result per input.  That loses IOC term ownership, so retain
                # the established field-local API in that case.
                if self._batched_multi_vjp_contract is None:
                    self._batched_multi_vjp_contract = "fused"
                pass

        for group in grouped_items:
            if len(group) < 2:
                raise AttributeError("batched RoboKots VJP group has fewer than two requests")
            ks = [int(key.k) for _index, key, _rhs, _state_ref in group]
            motions = [self._compose_motion(p, k=k) for k in ks]
            self._update_batched_dynamics(
                np.stack(motions, axis=0),
                p=p,
                time=time,
            )
            first_ref = group[0][3]
            total_joint_ref = self.adapter.as_total_joint_dynamics_state_ref(first_ref)
            refs: Any = first_ref if total_joint_ref is None else list(total_joint_ref.refs)
            rhs_batch = np.stack([rhs for _index, _key, rhs, _state_ref in group], axis=0)
            motion_grads = np.asarray(self.model.jacobian_transpose_mul(refs, rhs_batch), dtype=float)
            self._chain_batched_param_vjp_group(
                out=out,
                group=group,
                motions=motions,
                motion_grads=motion_grads,
            )
        return [np.asarray(value, dtype=float) for value in out]

    def param_jacobian_transpose_mul_many_fused(
        self,
        x_all: Array,
        requests: Sequence[tuple[StateKey, Array]],
        *,
        pack: Any = None,
        time: Any = None,
    ) -> Array:
        """Return the summed parameter VJP for compatible dynamics requests.

        RoboKots' ``jacobian_transpose_mul_many`` fuses its input pairs into
        one reverse dynamics pass and returns their *sum*.  That contract is
        ideal for a complete residual VJP, but cannot represent individual
        IOC-term columns; those continue to use
        :meth:`param_jacobian_transpose_mul_many` above.
        """
        if len(requests) < 2 or not self.batch_trajectory:
            raise AttributeError("fused batched trajectory VJP requires at least two requests")
        multi_vjp = getattr(self.model, "jacobian_transpose_mul_many", None)
        if not callable(multi_vjp):
            raise AttributeError("RoboKots model does not expose jacobian_transpose_mul_many")

        p = self._extract_q(np.asarray(x_all, dtype=float).reshape(-1), pack=pack)
        self._validate_trajectory_parameter_size(p)
        steps = self._expected_steps(time=time)
        prepared: list[tuple[StateKey, Array, Any, Any]] = []
        for key, rhs in requests:
            if not self._accept_required_key_for_traj(key, steps=steps):
                raise ValueError(f"KotsTrajectoryStateBuilder: invalid fused batched VJP key: {key!r}")
            if getattr(key, "dtype", None) != DTYPE_DYNAMICS:
                raise AttributeError("fused RoboKots VJP currently supports dynamics states only")
            route = self._route_for_key(key)
            entry = None if route is None else self._dispatch.get(route)
            if entry is None:
                raise AttributeError("fused RoboKots VJP requires a dynamics state handler")
            prepared.append((key, np.asarray(rhs, dtype=float), entry, self._state_ref(key, state_ref_field=entry.state_ref_field)))

        groups: dict[tuple[Any, ...], list[tuple[StateKey, Array, Any]]] = {}
        for key, rhs, entry, state_ref in prepared:
            signature = (
                entry.state_ref_field,
                getattr(getattr(key, "owner", None), "owner_type", None),
                getattr(getattr(key, "owner", None), "owner_name", None),
                getattr(key, "frame", None),
                rhs.shape[1:] if rhs.ndim >= 1 else (),
            )
            groups.setdefault(signature, []).append((key, rhs, state_ref))
        # A residual may contain more than one term for the same state field
        # and time.  RoboKots returns a sum, so combine those cotangents before
        # constructing its one request per state field.
        grouped_items: list[list[tuple[StateKey, Array, Any]]] = []
        for group in groups.values():
            by_k: dict[int, tuple[StateKey, Array, Any]] = {}
            for key, rhs, state_ref in group:
                k = int(key.k)
                if k in by_k:
                    prior_key, prior_rhs, prior_ref = by_k[k]
                    by_k[k] = (prior_key, np.asarray(prior_rhs + rhs, dtype=float), prior_ref)
                else:
                    by_k[k] = (key, np.asarray(rhs, dtype=float).copy(), state_ref)
            grouped_items.append([by_k[k] for k in sorted(by_k)])
        group_ks = [[int(key.k) for key, _rhs, _state_ref in group] for group in grouped_items]
        if not group_ks or any(ks != group_ks[0] for ks in group_ks[1:]):
            raise AttributeError("fused RoboKots VJP requires identical time grids for all state fields")

        motions = [self._compose_motion(p, k=k) for k in group_ks[0]]
        self._update_batched_dynamics(np.stack(motions, axis=0), p=p, time=time)
        backend_requests = []
        for group in grouped_items:
            first_ref = group[0][2]
            total_joint_ref = self.adapter.as_total_joint_dynamics_state_ref(first_ref)
            refs: Any = first_ref if total_joint_ref is None else list(total_joint_ref.refs)
            rhs_batch = np.stack([rhs for _key, rhs, _state_ref in group], axis=0)
            backend_requests.append((refs, rhs_batch))

        motion_grads = np.asarray(multi_vjp(backend_requests), dtype=float)
        if motion_grads.ndim < 2 or int(motion_grads.shape[0]) != len(motions):
            raise ValueError("RoboKots fused multi-VJP output must have one leading result per time step.")
        out = np.zeros((self.trajectory_map.p_dim,), dtype=float)
        for k, motion_grad in zip(group_ks[0], motion_grads, strict=True):
            mapped = self._trajectory_motion_gradient_transpose_at(k=k, motion_grad=np.asarray(motion_grad, dtype=float))
            if mapped is None:
                raise ValueError("RoboKots fused multi-VJP motion dimension does not match trajectory chain.")
            out += mapped
        return out

    def _chain_batched_param_vjp_group(
        self,
        *,
        out: list[Array | None],
        group: Sequence[tuple[int, StateKey, Array, Any]],
        motions: Sequence[Array],
        motion_grads: Array,
    ) -> None:
        """Map one RoboKots batched VJP result back to trajectory parameters."""
        grads = np.asarray(motion_grads, dtype=float)
        if grads.ndim < 2 or int(grads.shape[0]) != len(group):
            raise ValueError(
                "Batched RoboKots VJP output must have one leading result per requested time step."
            )
        for batch_index, (index, key, _rhs, _state_ref) in enumerate(group):
            motion_grad = np.asarray(grads[batch_index], dtype=float)
            mapped = self._trajectory_motion_gradient_transpose_at(k=int(key.k), motion_grad=motion_grad)
            if mapped is None:
                raise ValueError("Batched RoboKots VJP motion dimension does not match trajectory chain.")
            out[index] = mapped

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
