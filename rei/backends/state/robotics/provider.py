from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ....core.state_cache import StateKey
from ....core.state_schema import DTYPE_COORD, DTYPE_DYNAMICS, DTYPE_KINEMATICS, canonical_field_name
from ....core.trajectory import TrajectoryMap
from ..dispatch.template import BackendDispatchStateBuilder, DispatchHandler
from ..trajectory import (
    chain_param_jacobian,
    compose_interleaved_motion_and_jac,
    compose_stacked_motion_and_jac,
    TrajectoryStateBuilderMixin,
    validate_trajectory_derivative_maps,
)
from .binding_table import (
    BindingTable,
    DEFAULT_NAME_BINDING_OWNER_TYPES,
    RobotFieldBinding,
    register_robot_binding_table,
    register_robot_field_bindings,
    resolve_handler_ref,
    robot_field_bindings_from_table,
)
from .contract import assert_provider_contract, assert_trajectory_provider_contract

Array = np.ndarray
RobotUpdateFn = Callable[[Array, Any, Any], None]
TrajectoryRobotUpdateFn = Callable[[Array, Array, int, Any, Any], None]
RobotStateRefResolver = Callable[[StateKey, Any, Any], Any]
STATE_JACOBIAN_VAR = "state"


def _state_key_label(key: StateKey | None) -> str:
    if key is None:
        return "<unknown>"
    owner = getattr(key, "owner", None)
    owner_type = getattr(owner, "owner_type", None)
    owner_name = getattr(owner, "owner_name", None)
    return (
        f"k={getattr(key, 'k', None)!r}, "
        f"dtype={getattr(key, 'dtype', None)!r}, "
        f"owner_type={owner_type!r}, "
        f"owner_name={owner_name!r}, "
        f"field={getattr(key, 'field', None)!r}"
    )


def _handler_context(
    *,
    provider: str,
    dtype: str,
    owner_type: str,
    field: str,
    handler_name: str,
    key: StateKey | None = None,
    jacobian_wrt: str | None = None,
) -> str:
    parts = [
        f"{provider}[dtype={dtype!r}, owner_type={owner_type!r}, field={field!r}]",
        f"handler={handler_name}",
    ]
    if jacobian_wrt is not None:
        parts.append(f"jacobian_wrt={jacobian_wrt!r}")
    parts.append(f"key=({_state_key_label(key)})")
    return ": ".join(parts)


@dataclass(frozen=True)
class RobotStateRef:
    k: int
    owner_type: str | None
    owner_name: str
    dtype: str
    field: str
    frame: str | None = None
    rel_frame: str | None = None
    backend_ref: Any = None

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, str(name), default)


@dataclass(frozen=True)
class RobotFieldHandler:
    value_handler: DispatchHandler
    jac_handler: DispatchHandler | None = None
    jacobian_wrt: str | None = None


def _resolve_optional_callable_ref(
    owner: Any,
    ref: str | Callable[..., Any] | None,
    *,
    role: str,
) -> Callable[..., Any] | None:
    if ref is None:
        return None
    if callable(ref):
        return ref
    if not isinstance(ref, str) or ref == "":
        raise TypeError(f"{role}: reference must be a callable or non-empty method name.")
    if owner is None:
        raise ValueError(f"{role}: handler_owner is required when using method name {ref!r}.")
    fn = getattr(owner, ref, None)
    if not callable(fn):
        raise ValueError(f"{role}: handler_owner does not expose callable method {ref!r}.")
    return fn


class RoboticsStateProvider(BackendDispatchStateBuilder):
    """Generic robotics `build_state()` provider for custom backend libraries.

    This is the low-friction integration path for external robotics libraries:
      - `update_model(q, model, data)` syncs the backend model for the current q
      - each state field is provided by a `RobotFieldHandler`
      - optional `resolve_state_ref(key, model, data)` can build backend-native refs

    For more specialized backends, subclass `BackendDispatchStateBuilder` directly.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        q_var: str = "q",
        kinematics_owner_type: str = "link",
        dynamics_owner_type: str = "total_joint",
        kinematics_fields: Sequence[str] | None = None,
        dynamics_fields: Sequence[str] | None = None,
        kinematics_field_handlers: Mapping[str, RobotFieldHandler] | None = None,
        dynamics_field_handlers: Mapping[str, RobotFieldHandler] | None = None,
        update_model: RobotUpdateFn | None = None,
        resolve_state_ref: RobotStateRefResolver | None = None,
        register_joint_q: bool = True,
        allow_nonzero_k: bool = False,
        require_fields: bool = True,
        validate_handler_shapes: bool = True,
    ) -> None:
        super().__init__(
            model,
            data,
            q_var=q_var,
            allow_nonzero_k=allow_nonzero_k,
        )
        self.kinematics_owner_type = str(kinematics_owner_type)
        self.dynamics_owner_type = str(dynamics_owner_type)
        if self.kinematics_owner_type == "":
            raise ValueError("RoboticsStateProvider: kinematics_owner_type must be non-empty.")
        if self.dynamics_owner_type == "":
            raise ValueError("RoboticsStateProvider: dynamics_owner_type must be non-empty.")
        self.update_model = update_model
        self.resolve_state_ref = resolve_state_ref
        self.validate_handler_shapes = bool(validate_handler_shapes)
        self.field_to_jac: dict[tuple[str, str], str] = {}

        any_registered = False
        if register_joint_q:
            self.register_value_and_jac(
                dtype=DTYPE_COORD,
                owner_type="total_joint",
                field="q",
                value_handler=self._handle_joint_q_value,
                jac_handler=self._handle_joint_q_jac,
            )
            any_registered = True

        any_registered = self._register_field_group(
            dtype=DTYPE_KINEMATICS,
            owner_type=self.kinematics_owner_type,
            fields=kinematics_fields,
            field_handlers=kinematics_field_handlers,
            group_name="kinematics",
        ) or any_registered
        any_registered = self._register_field_group(
            dtype=DTYPE_DYNAMICS,
            owner_type=self.dynamics_owner_type,
            fields=dynamics_fields,
            field_handlers=dynamics_field_handlers,
            group_name="dynamics",
        ) or any_registered

        if require_fields and not any_registered:
            raise ValueError(
                "RoboticsStateProvider: no fields configured. Provide kinematics/dynamics "
                "field handlers or enable register_joint_q."
            )

    def _register_field_group(
        self,
        *,
        dtype: str,
        owner_type: str,
        fields: Sequence[str] | None,
        field_handlers: Mapping[str, RobotFieldHandler] | None,
        group_name: str,
    ) -> bool:
        handlers = {} if field_handlers is None else {canonical_field_name(str(k)): v for k, v in field_handlers.items()}
        selected_fields = list(handlers.keys()) if fields is None else [canonical_field_name(str(f)) for f in fields]
        if len(selected_fields) == 0:
            return False

        for field in selected_fields:
            if field == "":
                raise ValueError(f"RoboticsStateProvider: {group_name} field names must be non-empty.")
            spec = handlers.get(field, None)
            if spec is None:
                raise ValueError(
                    "RoboticsStateProvider: missing "
                    f"{group_name} field handler for {field!r}. Provide it in `{group_name}_field_handlers`."
                )
            if not isinstance(spec, RobotFieldHandler):
                raise TypeError(
                    "RoboticsStateProvider: field handler must be RobotFieldHandler. "
                    f"group={group_name!r}, field={field!r}, got={type(spec).__name__!r}."
                )
            self.register_robot_field(
                dtype=dtype,
                owner_type=owner_type,
                field=field,
                value_handler=spec.value_handler,
                jac_handler=spec.jac_handler,
                jacobian_wrt=spec.jacobian_wrt,
            )

        return True

    @classmethod
    def from_field_bindings(
        cls,
        model: Any,
        data: Any,
        *,
        field_bindings: Sequence[RobotFieldBinding],
        handler_owner: Any = None,
        q_var: str = "q",
        update_model: str | RobotUpdateFn | None = None,
        resolve_state_ref: str | RobotStateRefResolver | None = None,
        register_joint_q: bool = True,
        allow_nonzero_k: bool = False,
        validate_handler_shapes: bool = True,
    ) -> RoboticsStateProvider:
        """Build a provider from a simple list of key-to-method bindings."""

        provider = cls(
            model,
            data,
            q_var=q_var,
            update_model=_resolve_optional_callable_ref(
                handler_owner,
                update_model,
                role="RoboticsStateProvider.from_field_bindings(update_model)",
            ),
            resolve_state_ref=_resolve_optional_callable_ref(
                handler_owner,
                resolve_state_ref,
                role="RoboticsStateProvider.from_field_bindings(resolve_state_ref)",
            ),
            register_joint_q=register_joint_q,
            allow_nonzero_k=allow_nonzero_k,
            require_fields=False,
            validate_handler_shapes=validate_handler_shapes,
        )
        provider.register_field_bindings(field_bindings, handler_owner=handler_owner)
        if len(provider._dispatch) == 0:
            raise ValueError(
                "RoboticsStateProvider.from_field_bindings: no fields configured. "
                "Provide field_bindings or enable register_joint_q."
        )
        return provider

    @classmethod
    def from_binding_table(
        cls,
        model: Any,
        data: Any,
        *,
        bindings: BindingTable,
        handler_owner: Any = None,
        owner_types: Sequence[str] = DEFAULT_NAME_BINDING_OWNER_TYPES,
        q_var: str = "q",
        update_model: str | RobotUpdateFn | None = None,
        resolve_state_ref: str | RobotStateRefResolver | None = None,
        register_joint_q: bool = True,
        allow_nonzero_k: bool = False,
        validate_handler_shapes: bool = True,
    ) -> RoboticsStateProvider:
        """Build a provider from a dotted state-key to method-name table."""

        return cls.from_field_bindings(
            model,
            data,
            field_bindings=robot_field_bindings_from_table(
                bindings,
                owner_types=owner_types,
            ),
            handler_owner=handler_owner,
            q_var=q_var,
            update_model=update_model,
            resolve_state_ref=resolve_state_ref,
            register_joint_q=register_joint_q,
            allow_nonzero_k=allow_nonzero_k,
            validate_handler_shapes=validate_handler_shapes,
        )

    def register_field_bindings(
        self,
        field_bindings: Sequence[RobotFieldBinding],
        *,
        handler_owner: Any = None,
    ) -> None:
        for binding in field_bindings:
            if not isinstance(binding, RobotFieldBinding):
                raise TypeError(
                    "RoboticsStateProvider.register_field_bindings: entries must be RobotFieldBinding, "
                    f"got {type(binding).__name__!r}."
                )
            value_handler = resolve_handler_ref(
                handler_owner,
                binding.value,
                role="RobotFieldBinding.value",
                field=binding.field,
            )
            jac_handler = None
            if binding.jac is not None:
                jac_handler = resolve_handler_ref(
                    handler_owner,
                    binding.jac,
                    role="RobotFieldBinding.jac",
                    field=binding.field,
                )
            self.register_robot_field(
                dtype=binding.dtype,
                owner_type=binding.owner_type,
                field=binding.field,
                value_handler=value_handler,
                jac_handler=jac_handler,
                jacobian_wrt=binding.jacobian_wrt,
            )

    def register_robot_field(
        self,
        *,
        dtype: str,
        owner_type: str,
        field: str,
        value_handler: DispatchHandler,
        jac_handler: DispatchHandler | None = None,
        jacobian_wrt: str | None = None,
    ) -> tuple[str, str | None]:
        field_name = canonical_field_name(str(field))
        if field_name == "":
            raise ValueError("RoboticsStateProvider: field must be non-empty.")
        jacobian_wrt_name = self._normalize_jacobian_wrt(
            jacobian_wrt,
            dtype=str(dtype),
            owner_type=str(owner_type),
            field=field_name,
        )
        value_handler_wrapped = self._wrap_value_handler(
            dtype=str(dtype),
            owner_type=str(owner_type),
            field=field_name,
            handler=value_handler,
        )
        if jac_handler is None:
            self.register_handler(
                dtype=dtype,
                owner_type=owner_type,
                field=field_name,
                handler=value_handler_wrapped,
                state_ref_field=field_name,
            )
            return field_name, None

        jac_handler_wrapped = self._wrap_jac_handler(
            dtype=str(dtype),
            owner_type=str(owner_type),
            field=field_name,
            value_handler=value_handler_wrapped,
            jac_handler=jac_handler,
            jacobian_wrt=jacobian_wrt_name,
        )
        _value_name, jac_name = self.register_value_and_jac(
            dtype=dtype,
            owner_type=owner_type,
            field=field_name,
            value_handler=value_handler_wrapped,
            jac_handler=jac_handler_wrapped,
            jacobian_wrt=jacobian_wrt_name,
        )
        self.field_to_jac[(str(dtype), field_name)] = jac_name
        return field_name, jac_name

    def _normalize_jacobian_wrt(
        self,
        jacobian_wrt: str | None,
        *,
        dtype: str,
        owner_type: str,
        field: str,
    ) -> str | None:
        if jacobian_wrt is None:
            return None
        wrt = str(jacobian_wrt)
        if wrt == "":
            raise ValueError("RoboticsStateProvider: jacobian_wrt must be non-empty.")
        if wrt in (self.q_var, STATE_JACOBIAN_VAR):
            return wrt
        raise ValueError(
            "RoboticsStateProvider: unsupported jacobian_wrt for field handler. "
            f"dtype={dtype!r}, owner_type={owner_type!r}, field={field!r}, "
            f"expected {self.q_var!r} or {STATE_JACOBIAN_VAR!r}, got {wrt!r}."
        )

    @staticmethod
    def _as_value_array(raw: Any, *, where: str) -> Array:
        try:
            arr = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: value handler must return numeric data.") from exc
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr.reshape(-1)

    @staticmethod
    def _as_jacobian_array(raw: Any, *, where: str) -> Array:
        try:
            arr = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: Jacobian handler must return numeric data.") from exc
        if arr.ndim != 2:
            raise ValueError(
                f"{where}: Jacobian handler must return a 2D array. "
                f"Expected shape=(m, n), actual shape={arr.shape}."
            )
        return arr

    def _wrap_value_handler(
        self,
        *,
        dtype: str,
        owner_type: str,
        field: str,
        handler: DispatchHandler,
    ) -> DispatchHandler:
        def _wrapped(q: Array, key: StateKey, state_ref: Any) -> Array:
            where = _handler_context(
                provider=type(self).__name__,
                dtype=dtype,
                owner_type=owner_type,
                field=field,
                handler_name="value_handler",
                key=key,
            )
            value = self._as_value_array(handler(q, key, state_ref), where=where)
            if self.validate_handler_shapes and value.size == 0:
                raise ValueError(
                    f"{where}: value shape mismatch. Expected non-empty vector shape=(m,), "
                    f"actual shape={value.shape}."
                )
            return value

        return _wrapped

    def _expected_jacobian_columns(
        self,
        *,
        q: Array,
        jacobian_wrt: str | None,
    ) -> int | None:
        wrt = self.q_var if jacobian_wrt is None else str(jacobian_wrt)
        if wrt == self.q_var:
            return int(np.asarray(q, dtype=float).reshape(-1).size)
        if wrt == STATE_JACOBIAN_VAR:
            return None
        return None

    def _wrap_jac_handler(
        self,
        *,
        dtype: str,
        owner_type: str,
        field: str,
        value_handler: DispatchHandler,
        jac_handler: DispatchHandler,
        jacobian_wrt: str | None,
    ) -> DispatchHandler:
        def _wrapped(q: Array, key: StateKey, state_ref: Any) -> Array:
            where = _handler_context(
                provider=type(self).__name__,
                dtype=dtype,
                owner_type=owner_type,
                field=field,
                handler_name="jac_handler",
                key=key,
                jacobian_wrt=self.q_var if jacobian_wrt is None else str(jacobian_wrt),
            )
            J = self._as_jacobian_array(jac_handler(q, key, state_ref), where=where)
            if not self.validate_handler_shapes:
                return J

            value = self._as_value_array(value_handler(q, key, state_ref), where=where)
            if J.shape[0] != value.size:
                raise ValueError(
                    f"{where}: Jacobian row mismatch. "
                    f"Expected shape=({value.size}, *), actual shape={J.shape}; "
                    f"value shape={value.shape}."
                )

            expected_cols = self._expected_jacobian_columns(q=q, jacobian_wrt=jacobian_wrt)
            if expected_cols is not None and J.shape[1] != expected_cols:
                raise ValueError(
                    f"{where}: Jacobian column mismatch. "
                    f"Expected shape=({value.size}, {expected_cols}), actual shape={J.shape}."
                )
            return J

        return _wrapped

    def _update_kinematics(self, q: Array) -> None:
        if self.update_model is None:
            return
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        self.update_model(q_vec, self.model, self.data)

    def _resolve_state_ref(self, key: StateKey) -> Any:
        if self.resolve_state_ref is not None:
            return self.resolve_state_ref(key, self.model, self.data)

        owner = getattr(key, "owner", None)
        owner_type = getattr(owner, "owner_type", None)
        owner_name = getattr(owner, "owner_name", None)
        if not isinstance(owner_name, str) or owner_name == "":
            raise ValueError("RoboticsStateProvider expects non-empty owner_name in key, got: " f"{key!r}")

        return RobotStateRef(
            k=int(getattr(key, "k", 0)),
            owner_type=None if owner_type is None else str(owner_type),
            owner_name=owner_name,
            dtype=str(getattr(key, "dtype", "")),
            field=str(getattr(key, "field", "")),
            frame=getattr(key, "frame", None),
            rel_frame=getattr(key, "rel_frame", None),
        )

    def _handle_joint_q_value(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        return np.asarray(q, dtype=float).reshape(-1).copy()

    def _handle_joint_q_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        n = int(np.asarray(q, dtype=float).reshape(-1).size)
        return np.eye(n, dtype=float)


class TrajectoryRoboticsStateProvider(TrajectoryStateBuilderMixin, RoboticsStateProvider):
    """Trajectory-parameterized robotics provider for custom backend libraries.

    The decision variable is `p_var`. At each requested time index, this provider:
      - evaluates `q(k) = trajectory_map.q_at(p, k)`
      - optionally builds a motion vector (`q`, stacked derivatives, or interleaved derivatives)
      - updates the backend model
      - chains state Jacobians to parameter Jacobians when needed

    Field Jacobian handlers should normally return derivatives with respect to
    backend state/motion. Set `RobotFieldHandler(jacobian_wrt=p_var)` only when
    the handler already returns a parameter-space Jacobian.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        trajectory_map: TrajectoryMap,
        trajectory_derivative_maps: Mapping[int, TrajectoryMap] | None = None,
        p_var: str = "p",
        kinematics_owner_type: str = "link",
        dynamics_owner_type: str = "total_joint",
        kinematics_fields: Sequence[str] | None = None,
        dynamics_fields: Sequence[str] | None = None,
        kinematics_field_handlers: Mapping[str, RobotFieldHandler] | None = None,
        dynamics_field_handlers: Mapping[str, RobotFieldHandler] | None = None,
        update_model: RobotUpdateFn | None = None,
        update_motion_model: TrajectoryRobotUpdateFn | None = None,
        resolve_state_ref: RobotStateRefResolver | None = None,
        register_joint_q: bool = True,
        motion_layout: str = "q",
        motion_order: int | None = None,
        derivative_orders: Sequence[int] = (0, 1, 2),
        validate_handler_shapes: bool = True,
        require_fields: bool = True,
    ) -> None:
        self.trajectory_map = trajectory_map
        self.trajectory_derivative_maps: dict[int, TrajectoryMap] = {0: trajectory_map}
        if trajectory_derivative_maps is not None:
            for order_raw, traj in trajectory_derivative_maps.items():
                order = int(order_raw)
                if order < 0:
                    raise ValueError(
                        f"TrajectoryRoboticsStateProvider: derivative order must be >= 0, got {order}."
                    )
                self.trajectory_derivative_maps[order] = traj
        validate_trajectory_derivative_maps(
            self.trajectory_derivative_maps,
            error_prefix="TrajectoryRoboticsStateProvider",
        )

        self.p_var = str(p_var)
        if self.p_var == "":
            raise ValueError("TrajectoryRoboticsStateProvider: p_var must be non-empty.")
        self.update_motion_model = update_motion_model
        self.motion_layout = str(motion_layout).strip().lower()
        if self.motion_layout not in ("q", "stacked", "interleaved"):
            raise ValueError(
                "TrajectoryRoboticsStateProvider: motion_layout must be one of "
                "'q', 'stacked', or 'interleaved'."
            )
        self.motion_order = None if motion_order is None else int(motion_order)
        if self.motion_order is not None and self.motion_order <= 0:
            raise ValueError("TrajectoryRoboticsStateProvider: motion_order must be > 0 when provided.")
        self.derivative_orders = tuple(int(o) for o in derivative_orders)
        if any(o < 0 for o in self.derivative_orders):
            raise ValueError("TrajectoryRoboticsStateProvider: derivative_orders must be >= 0.")
        if self.motion_layout == "stacked" and len(self.derivative_orders) == 0:
            raise ValueError("TrajectoryRoboticsStateProvider: stacked motion requires derivative_orders.")

        super().__init__(
            model,
            data,
            q_var=self.p_var,
            kinematics_owner_type=kinematics_owner_type,
            dynamics_owner_type=dynamics_owner_type,
            kinematics_fields=kinematics_fields,
            dynamics_fields=dynamics_fields,
            kinematics_field_handlers=self._default_state_jacobian_wrt(kinematics_field_handlers),
            dynamics_field_handlers=self._default_state_jacobian_wrt(dynamics_field_handlers),
            update_model=update_model,
            resolve_state_ref=resolve_state_ref,
            register_joint_q=False,
            allow_nonzero_k=True,
            require_fields=False,
            validate_handler_shapes=validate_handler_shapes,
        )
        if register_joint_q:
            self.register_value_and_jac(
                dtype=DTYPE_COORD,
                owner_type="total_joint",
                field="q",
                value_handler=self._handle_joint_q_value,
                jac_handler=self._handle_joint_q_jac,
            )
        if require_fields and len(self._dispatch) == 0:
            raise ValueError(
                "TrajectoryRoboticsStateProvider: no fields configured. Provide kinematics/dynamics "
                "field handlers or enable register_joint_q."
            )

    @classmethod
    def from_field_bindings(
        cls,
        model: Any,
        data: Any,
        *,
        trajectory_map: TrajectoryMap,
        field_bindings: Sequence[RobotFieldBinding],
        handler_owner: Any = None,
        trajectory_derivative_maps: Mapping[int, TrajectoryMap] | None = None,
        p_var: str = "p",
        update_model: str | RobotUpdateFn | None = None,
        update_motion_model: str | TrajectoryRobotUpdateFn | None = None,
        resolve_state_ref: str | RobotStateRefResolver | None = None,
        register_joint_q: bool = True,
        motion_layout: str = "q",
        motion_order: int | None = None,
        derivative_orders: Sequence[int] = (0, 1, 2),
        validate_handler_shapes: bool = True,
    ) -> TrajectoryRoboticsStateProvider:
        """Build a trajectory provider from a simple list of key-to-method bindings."""

        provider = cls(
            model,
            data,
            trajectory_map=trajectory_map,
            trajectory_derivative_maps=trajectory_derivative_maps,
            p_var=p_var,
            update_model=_resolve_optional_callable_ref(
                handler_owner,
                update_model,
                role="TrajectoryRoboticsStateProvider.from_field_bindings(update_model)",
            ),
            update_motion_model=_resolve_optional_callable_ref(
                handler_owner,
                update_motion_model,
                role="TrajectoryRoboticsStateProvider.from_field_bindings(update_motion_model)",
            ),
            resolve_state_ref=_resolve_optional_callable_ref(
                handler_owner,
                resolve_state_ref,
                role="TrajectoryRoboticsStateProvider.from_field_bindings(resolve_state_ref)",
            ),
            register_joint_q=register_joint_q,
            motion_layout=motion_layout,
            motion_order=motion_order,
            derivative_orders=derivative_orders,
            validate_handler_shapes=validate_handler_shapes,
            require_fields=False,
        )
        provider.register_field_bindings(
            cls._default_state_jacobian_wrt_for_bindings(field_bindings),
            handler_owner=handler_owner,
        )
        if len(provider._dispatch) == 0:
            raise ValueError(
                "TrajectoryRoboticsStateProvider.from_field_bindings: no fields configured. "
                "Provide field_bindings or enable register_joint_q."
            )
        return provider

    @classmethod
    def from_binding_table(
        cls,
        model: Any,
        data: Any,
        *,
        trajectory_map: TrajectoryMap,
        bindings: BindingTable,
        handler_owner: Any = None,
        owner_types: Sequence[str] = DEFAULT_NAME_BINDING_OWNER_TYPES,
        trajectory_derivative_maps: Mapping[int, TrajectoryMap] | None = None,
        p_var: str = "p",
        update_model: str | RobotUpdateFn | None = None,
        update_motion_model: str | TrajectoryRobotUpdateFn | None = None,
        resolve_state_ref: str | RobotStateRefResolver | None = None,
        register_joint_q: bool = True,
        motion_layout: str = "q",
        motion_order: int | None = None,
        derivative_orders: Sequence[int] = (0, 1, 2),
        validate_handler_shapes: bool = True,
    ) -> TrajectoryRoboticsStateProvider:
        """Build a trajectory provider from a dotted state-key to method-name table."""

        return cls.from_field_bindings(
            model,
            data,
            trajectory_map=trajectory_map,
            field_bindings=robot_field_bindings_from_table(
                bindings,
                owner_types=owner_types,
                default_jacobian_wrt=STATE_JACOBIAN_VAR,
            ),
            handler_owner=handler_owner,
            trajectory_derivative_maps=trajectory_derivative_maps,
            p_var=p_var,
            update_model=update_model,
            update_motion_model=update_motion_model,
            resolve_state_ref=resolve_state_ref,
            register_joint_q=register_joint_q,
            motion_layout=motion_layout,
            motion_order=motion_order,
            derivative_orders=derivative_orders,
            validate_handler_shapes=validate_handler_shapes,
        )

    @staticmethod
    def _default_state_jacobian_wrt_for_bindings(
        bindings: Sequence[RobotFieldBinding],
    ) -> tuple[RobotFieldBinding, ...]:
        out: list[RobotFieldBinding] = []
        for binding in bindings:
            if not isinstance(binding, RobotFieldBinding):
                raise TypeError(
                    "TrajectoryRoboticsStateProvider.from_field_bindings: entries must be RobotFieldBinding, "
                    f"got {type(binding).__name__!r}."
                )
            if binding.jac is None or binding.jacobian_wrt is not None:
                out.append(binding)
                continue
            out.append(
                RobotFieldBinding(
                    dtype=binding.dtype,
                    owner_type=binding.owner_type,
                    field=binding.field,
                    value=binding.value,
                    jac=binding.jac,
                    jacobian_wrt=STATE_JACOBIAN_VAR,
                )
            )
        return tuple(out)

    @staticmethod
    def _default_state_jacobian_wrt(
        handlers: Mapping[str, RobotFieldHandler] | None,
    ) -> Mapping[str, RobotFieldHandler] | None:
        if handlers is None:
            return None
        out: dict[str, RobotFieldHandler] = {}
        for field, spec in handlers.items():
            if not isinstance(spec, RobotFieldHandler) or spec.jac_handler is None or spec.jacobian_wrt is not None:
                out[str(field)] = spec
                continue
            out[str(field)] = RobotFieldHandler(
                value_handler=spec.value_handler,
                jac_handler=spec.jac_handler,
                jacobian_wrt=STATE_JACOBIAN_VAR,
            )
        return out

    def _compose_motion_and_jac(self, p: Array, *, k: int) -> tuple[Array, Array]:
        p_vec = np.asarray(p, dtype=float).reshape(-1)
        if self.motion_layout == "q":
            return (
                np.asarray(self.trajectory_map.q_at(p_vec, k), dtype=float).reshape(-1),
                np.asarray(self.trajectory_map.dqdp_at(k), dtype=float),
            )

        if self.motion_layout == "stacked":
            return compose_stacked_motion_and_jac(
                p_vec,
                trajectory_map=self.trajectory_map,
                trajectory_derivative_maps=self.trajectory_derivative_maps,
                derivative_orders=self.derivative_orders,
                k=k,
                error_prefix="TrajectoryRoboticsStateProvider",
            )

        order = self.motion_order
        if order is None:
            order = max(self.trajectory_derivative_maps.keys()) + 1
        return compose_interleaved_motion_and_jac(
            p_vec,
            trajectory_map=self.trajectory_map,
            trajectory_derivative_maps=self.trajectory_derivative_maps,
            order=int(order),
            k=k,
            error_prefix="TrajectoryRoboticsStateProvider",
        )

    def _compose_motion(self, p: Array, *, k: int) -> Array:
        p_vec = np.asarray(p, dtype=float).reshape(-1)
        if self.motion_layout == "q":
            return np.asarray(self.trajectory_map.q_at(p_vec, k), dtype=float).reshape(-1)

        dof = int(self.trajectory_map.q_dim)
        if self.motion_layout == "stacked":
            parts: list[Array] = []
            for deriv_order in self.derivative_orders:
                traj = self.trajectory_derivative_maps.get(int(deriv_order), None)
                q_r = (
                    np.zeros((dof,), dtype=float)
                    if traj is None
                    else np.asarray(traj.q_at(p_vec, k), dtype=float).reshape(-1)
                )
                if q_r.size != dof:
                    raise ValueError(
                        "TrajectoryRoboticsStateProvider: derivative map q size mismatch. "
                        f"order={deriv_order}, expected {dof}, got {q_r.size}."
                    )
                parts.append(q_r)
            return np.concatenate(parts, axis=0)

        order = self.motion_order
        if order is None:
            order = max(self.trajectory_derivative_maps.keys()) + 1
        order_i = int(order)
        motion = np.zeros((dof * order_i,), dtype=float)
        for deriv_order, traj in self.trajectory_derivative_maps.items():
            deriv_order_i = int(deriv_order)
            if deriv_order_i >= order_i:
                continue
            q_r = np.asarray(traj.q_at(p_vec, k), dtype=float).reshape(-1)
            if q_r.size != dof:
                raise ValueError(
                    "TrajectoryRoboticsStateProvider: derivative map q size mismatch. "
                    f"order={deriv_order_i}, expected {dof}, got {q_r.size}."
                )
            motion[deriv_order_i::order_i] = q_r
        return motion

    def _update_trajectory_step(self, *, k: int, q_k: Array, motion_k: Array) -> None:
        q_vec = np.asarray(q_k, dtype=float).reshape(-1)
        motion_vec = np.asarray(motion_k, dtype=float).reshape(-1)
        if self.update_motion_model is not None:
            self.update_motion_model(q_vec, motion_vec, int(k), self.model, self.data)
            return
        self._update_kinematics(q_vec)

    def _handle_joint_q_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del q, state_ref
        k = int(getattr(key, "k", 0))
        return np.asarray(self.trajectory_map.dqdp_at(k), dtype=float)

    def _chain_param_jac(
        self,
        J_raw: Array,
        *,
        key: StateKey,
        jacobian_wrt: str | None,
        dqdp_k: Array,
        dmotiondp_k: Array,
    ) -> Array:
        return chain_param_jacobian(
            J_raw,
            q_var=self.p_var,
            state_jacobian_var=STATE_JACOBIAN_VAR,
            key=key,
            jacobian_wrt=jacobian_wrt,
            dqdp=dqdp_k,
            dmotiondp=dmotiondp_k,
            error_prefix="TrajectoryRoboticsStateProvider",
        )


__all__ = [
    "RobotFieldHandler",
    "RobotFieldBinding",
    "RobotStateRef",
    "RobotStateRefResolver",
    "RobotUpdateFn",
    "RoboticsStateProvider",
    "STATE_JACOBIAN_VAR",
    "TrajectoryRobotUpdateFn",
    "TrajectoryRoboticsStateProvider",
    "assert_provider_contract",
    "assert_trajectory_provider_contract",
    "register_robot_binding_table",
    "register_robot_field_bindings",
    "robot_field_bindings_from_table",
]
