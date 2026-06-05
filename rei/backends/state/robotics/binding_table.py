from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ....core.state_schema import DTYPE_COORD, DTYPE_DYNAMICS, DTYPE_KINEMATICS, canonical_field_name
from ..dispatch.template import BackendDispatchStateBuilder, DispatchHandler

HandlerRef = str | DispatchHandler
BindingTable = Mapping[str, HandlerRef] | str
DEFAULT_NAME_BINDING_OWNER_TYPES = ("total_joint", "link", "joint")


@dataclass(frozen=True)
class RobotFieldBinding:
    """Declarative field binding for simple custom robotics backends.

    `value` and `jac` can be callables or method names on a handler object.
    """

    dtype: str
    owner_type: str
    field: str
    value: HandlerRef
    jac: HandlerRef | None = None
    jacobian_wrt: str | None = None


def resolve_handler_ref(
    owner: Any,
    ref: HandlerRef,
    *,
    role: str,
    field: str | None = None,
) -> DispatchHandler:
    if callable(ref):
        return ref
    if not isinstance(ref, str) or ref == "":
        raise TypeError(f"{role}: handler reference must be a callable or non-empty method name.")
    if owner is None:
        raise ValueError(f"{role}: handler_owner is required when using method name {ref!r}.")
    fn = getattr(owner, ref, None)
    if not callable(fn):
        field_msg = "" if field is None else f" for field {field!r}"
        raise ValueError(f"{role}: handler_owner does not expose callable method {ref!r}{field_msg}.")
    return fn


def _parse_binding_table_text(table: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line_no, raw_line in enumerate(table.splitlines(), start=1):
        line = raw_line.strip()
        if line == "" or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(
                "binding table text lines must use `<state-key> = <method-name>`. "
                f"Invalid line {line_no}: {raw_line!r}."
            )
        key, value = (part.strip() for part in line.split("=", 1))
        if key == "" or value == "":
            raise ValueError(
                "binding table text lines require both a state key and method name. "
                f"Invalid line {line_no}: {raw_line!r}."
            )
        if key in out:
            raise ValueError(f"duplicate binding table key {key!r} on line {line_no}.")
        out[key] = value
    return out


def _parse_binding_table_key(
    key: str,
    *,
    owner_types: Sequence[str],
) -> tuple[str, str, str, str | None]:
    key_str = str(key).strip()
    if key_str == "":
        raise ValueError("binding table keys must be non-empty.")
    if "." not in key_str:
        raise ValueError(
            f"invalid robotics binding table key {key_str!r}. Expected "
            "`<dtype>.<owner_type>.<field>` or `<dtype>.<owner_type>.<field>.J_<var>`."
        )

    parts = key_str.split(".")
    if len(parts) not in (3, 4):
        raise ValueError(
            f"invalid robotics binding table key {key_str!r}. Expected "
            "`<dtype>.<owner_type>.<field>` or `<dtype>.<owner_type>.<field>.J_<var>`."
        )
    dtype, owner_type, field = (part.strip() for part in parts[:3])
    if dtype == "" or owner_type == "" or field == "":
        raise ValueError(f"invalid robotics binding table key {key_str!r}: empty key segment.")
    owner_type_names = {str(v) for v in owner_types}
    if owner_type not in owner_type_names:
        allowed_owners = ", ".join(repr(v) for v in sorted(owner_type_names))
        raise ValueError(
            f"invalid robotics binding table key {key_str!r}: unknown owner type "
            f"{owner_type!r}. Known owner types: {allowed_owners}."
        )
    if dtype not in (DTYPE_KINEMATICS, DTYPE_DYNAMICS, DTYPE_COORD):
        raise ValueError(
            f"invalid robotics binding table key {key_str!r}: dtype must be one of "
            f"{DTYPE_KINEMATICS!r}, {DTYPE_DYNAMICS!r}, {DTYPE_COORD!r}."
        )

    field_name = canonical_field_name(field)
    if len(parts) == 3:
        return dtype, owner_type, field_name, None

    jac_segment = parts[3].strip()
    if not jac_segment.startswith("J_") or jac_segment == "J_":
        raise ValueError(
            f"invalid robotics binding table key {key_str!r}. Jacobian segment must be `J_<var>`."
        )
    return dtype, owner_type, field_name, jac_segment[2:]


def robot_field_bindings_from_table(
    bindings: BindingTable,
    *,
    owner_types: Sequence[str] = DEFAULT_NAME_BINDING_OWNER_TYPES,
    default_jacobian_wrt: str | None = None,
) -> tuple[RobotFieldBinding, ...]:
    """Convert a dotted binding table into `RobotFieldBinding` objects."""

    if isinstance(bindings, str):
        table_methods: Mapping[str, HandlerRef] = _parse_binding_table_text(bindings)
    else:
        table_methods = bindings

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    for key, ref in table_methods.items():
        dtype, owner_type, field, jacobian_wrt = _parse_binding_table_key(str(key), owner_types=owner_types)
        route = (dtype, owner_type, field)
        if route not in grouped:
            grouped[route] = {"value": None, "jac": None, "jacobian_wrt": default_jacobian_wrt}
            order.append(route)
        entry = grouped[route]
        if jacobian_wrt is None:
            if entry["value"] is not None:
                raise ValueError(f"duplicate value binding for {key!r}.")
            entry["value"] = ref
            continue
        if entry["jac"] is not None:
            raise ValueError(f"duplicate Jacobian binding for {key!r}.")
        entry["jac"] = ref
        entry["jacobian_wrt"] = jacobian_wrt

    out: list[RobotFieldBinding] = []
    for dtype, owner_type, field in order:
        entry = grouped[(dtype, owner_type, field)]
        if entry["value"] is None:
            raise ValueError(
                "binding tables require a value binding for every Jacobian binding. "
                f"Missing value for {dtype}.{owner_type}.{field}."
            )
        out.append(
            RobotFieldBinding(
                dtype=dtype,
                owner_type=owner_type,
                field=field,
                value=entry["value"],
                jac=entry["jac"],
                jacobian_wrt=entry["jacobian_wrt"],
            )
        )
    return tuple(out)


def register_robot_field_bindings(
    builder: BackendDispatchStateBuilder,
    field_bindings: Sequence[RobotFieldBinding],
    *,
    handler_owner: Any = None,
) -> dict[tuple[str, str, str], str | None]:
    """Register `RobotFieldBinding` entries on a dispatch-style state builder."""

    out: dict[tuple[str, str, str], str | None] = {}
    for binding in field_bindings:
        if not isinstance(binding, RobotFieldBinding):
            raise TypeError(
                "register_robot_field_bindings: entries must be RobotFieldBinding, "
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
        _value_name, jac_name = builder.register_value_and_jac(
            dtype=binding.dtype,
            owner_type=binding.owner_type,
            field=binding.field,
            value_handler=value_handler,
            jac_handler=jac_handler,
            jacobian_wrt=binding.jacobian_wrt,
        )
        out[(binding.dtype, binding.owner_type, binding.field)] = jac_name if binding.jac is not None else None
    return out


def register_robot_binding_table(
    builder: BackendDispatchStateBuilder,
    bindings: BindingTable,
    *,
    handler_owner: Any = None,
    owner_types: Sequence[str] = DEFAULT_NAME_BINDING_OWNER_TYPES,
    default_jacobian_wrt: str | None = None,
) -> dict[tuple[str, str, str], str | None]:
    """Register a dotted state-key to method-name table on a dispatch builder."""

    return register_robot_field_bindings(
        builder,
        robot_field_bindings_from_table(
            bindings,
            owner_types=owner_types,
            default_jacobian_wrt=default_jacobian_wrt,
        ),
        handler_owner=handler_owner,
    )


__all__ = [
    "BindingTable",
    "DEFAULT_NAME_BINDING_OWNER_TYPES",
    "HandlerRef",
    "RobotFieldBinding",
    "register_robot_binding_table",
    "register_robot_field_bindings",
    "resolve_handler_ref",
    "robot_field_bindings_from_table",
]
