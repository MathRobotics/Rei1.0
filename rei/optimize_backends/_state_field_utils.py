from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..core.state_schema import canonical_dtype_name, canonical_field_name, split_jac_field
from ..optimize.dsl.dsl_ops import iter_nodes
from ..optimize.runtime import NLSRuntime


def base_field_name(field: str) -> str:
    field_name = canonical_field_name(str(field))
    try:
        base, _var = split_jac_field(field_name)
    except ValueError:
        return field_name
    return base


def canonicalize_unique_fields(
    fields: Sequence[str] | None,
    *,
    where: str,
    param_name: str = "fields",
) -> tuple[str, ...] | None:
    if fields is None:
        return None

    out: list[str] = []
    seen: set[str] = set()
    for raw in fields:
        field = canonical_field_name(str(raw).strip())
        if field == "" or field in seen:
            continue
        seen.add(field)
        out.append(field)

    if len(out) == 0:
        raise ValueError(f"{where}: {param_name} must be non-empty when provided.")
    return tuple(out)


def _explicit_get_state_jac_field_bases(node: Mapping[str, Any]) -> list[str]:
    jac_entries: list[Mapping[str, Any]] = []

    jac_dsl = node.get("jac", None)
    if jac_dsl is not None:
        if not isinstance(jac_dsl, Mapping):
            raise ValueError("get_state: jac must be a dict when provided.")
        jac_entries.append(jac_dsl)

    jacs_dsl = node.get("jacs", None)
    if jacs_dsl is not None:
        if not isinstance(jacs_dsl, list):
            raise ValueError("get_state: jacs must be a list[dict] when provided.")
        for i, item in enumerate(jacs_dsl):
            if not isinstance(item, Mapping):
                raise ValueError(f"get_state: jacs[{i}] must be a dict, got {type(item).__name__}.")
            jac_entries.append(item)

    out: list[str] = []
    for entry in jac_entries:
        field_raw = entry.get("field", None)
        if field_raw is None:
            continue
        out.append(base_field_name(str(field_raw)))
    return out


def required_base_fields_in_order_from_dsl(
    *,
    dsl: Mapping[str, Any],
    dtype: str,
    owner_type: str,
) -> tuple[list[str], set[str]]:
    requested_fields: list[str] = []
    requested_seen: set[str] = set()
    unsupported_owner_types: set[str] = set()
    dtype_name = str(dtype)
    owner_type_name = str(owner_type)

    for term in dsl.get("terms", []) or []:
        if not isinstance(term, Mapping):
            continue
        expr = term.get("expr", None)
        for node in iter_nodes(expr):
            if node.get("type", None) != "get_state":
                continue
            key = node.get("key", None)
            if not isinstance(key, Mapping):
                continue

            key_dtype_name = canonical_dtype_name(str(key.get("dtype", "")))
            if key_dtype_name != dtype_name:
                continue

            key_owner_type = str(key.get("owner_type", ""))
            if key_owner_type != owner_type_name:
                unsupported_owner_types.add(key_owner_type)
                continue

            field = base_field_name(str(key.get("field", "")))
            if field not in requested_seen:
                requested_seen.add(field)
                requested_fields.append(field)

            for jac_field_base in _explicit_get_state_jac_field_bases(node):
                if jac_field_base in requested_seen:
                    continue
                requested_seen.add(jac_field_base)
                requested_fields.append(jac_field_base)

    return requested_fields, unsupported_owner_types


def required_base_fields_in_order(
    *,
    runtime: NLSRuntime,
    dtype: str,
    owner_type: str,
) -> tuple[list[str], set[str]]:
    requested_fields: list[str] = []
    requested_seen: set[str] = set()
    unsupported_owner_types: set[str] = set()
    dtype_name = str(dtype)
    owner_type_name = str(owner_type)

    for key in runtime.required:
        if getattr(key, "dtype", None) != dtype_name:
            continue
        owner = getattr(key, "owner", None)
        key_owner_type = getattr(owner, "owner_type", None)
        if key_owner_type != owner_type_name:
            unsupported_owner_types.add(str(key_owner_type))
            continue
        field = base_field_name(str(getattr(key, "field", "")))
        if field in requested_seen:
            continue
        requested_seen.add(field)
        requested_fields.append(field)
    return requested_fields, unsupported_owner_types


def required_base_fields(
    *,
    runtime: NLSRuntime,
    dtype: str,
    owner_type: str,
) -> tuple[set[str], set[str]]:
    requested_fields, unsupported_owner_types = required_base_fields_in_order(
        runtime=runtime,
        dtype=dtype,
        owner_type=owner_type,
    )
    return set(requested_fields), unsupported_owner_types


def registered_base_fields(
    *,
    builder: Any,
    dtype: str,
    owner_type: str,
) -> set[str]:
    fields: set[str] = set()
    for field in builder.registered_route_fields(dtype=dtype, owner_type=owner_type):
        fields.add(base_field_name(field))
    return fields


def validate_runtime_field_coverage(
    *,
    runtime: NLSRuntime,
    builder: Any,
    dtype: str,
    owner_type: str,
    error_prefix: str,
    builder_name: str,
    missing_hint: str,
) -> None:
    requested_fields, unsupported_owner_types = required_base_fields(
        runtime=runtime,
        dtype=dtype,
        owner_type=owner_type,
    )
    if unsupported_owner_types:
        unsupported = ", ".join(sorted(unsupported_owner_types))
        raise ValueError(
            f"{error_prefix}: DSL contains {dtype} keys with unsupported owner_type(s): "
            f"{unsupported}. Supported owner_type is {owner_type!r}."
        )
    if len(requested_fields) == 0:
        return

    registered_fields = registered_base_fields(
        builder=builder,
        dtype=dtype,
        owner_type=owner_type,
    )
    missing_fields = sorted(requested_fields - registered_fields)
    if len(missing_fields) == 0:
        return

    requested_str = ", ".join(sorted(requested_fields))
    registered_str = ", ".join(sorted(registered_fields)) if len(registered_fields) > 0 else "<none>"
    missing_str = ", ".join(missing_fields)
    raise ValueError(
        f"{error_prefix}: DSL requests {dtype} field(s) that are not registered in "
        f"{builder_name}. "
        f"Missing: {missing_str}. Requested: {requested_str}. Registered: {registered_str}. "
        f"{missing_hint}"
    )


__all__ = [
    "base_field_name",
    "canonicalize_unique_fields",
    "required_base_fields_in_order",
    "required_base_fields_in_order_from_dsl",
    "required_base_fields",
    "registered_base_fields",
    "validate_runtime_field_coverage",
]
