from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OptValAlias:
    name: str
    canonical_name: str
    description: str = ""


DEFAULT_OPT_VAL_ALIASES: dict[str, OptValAlias] = {
    "joint_angles": OptValAlias(
        name="joint_angles",
        canonical_name="q",
        description="Robot joint angle vector.",
    ),
    "trajectory_params": OptValAlias(
        name="trajectory_params",
        canonical_name="p",
        description="Trajectory parameter vector.",
    ),
    "camera_params": OptValAlias(
        name="camera_params",
        canonical_name="theta",
        description="Camera parameter vector.",
    ),
    "system_params": OptValAlias(
        name="system_params",
        canonical_name="params",
        description="Generic physical system parameter vector.",
    ),
}


@dataclass(frozen=True)
class OptValResolution:
    variables: tuple[dict[str, Any], ...]
    aliases: dict[str, str]

    def resolve_var(self, name: Any) -> str:
        text = str(name)
        return self.aliases.get(text, text)


@dataclass(frozen=True)
class QuantityAlias:
    name: str
    source: str
    var: str | None = None
    derivative_order: int | None = None
    derivative_wrt: str | None = None
    owner_type: str | None = None
    owner_name: str | None = None
    dtype: str | None = None
    field: str | None = None
    description: str = ""


DEFAULT_QUANTITY_ALIASES: dict[str, QuantityAlias] = {
    "joint_angles": QuantityAlias(
        name="joint_angles",
        source="trajectory",
        var="trajectory_params",
        derivative_order=0,
        description="Joint angle trajectory.",
    ),
    "joint_velocities": QuantityAlias(
        name="joint_velocities",
        source="trajectory",
        var="trajectory_params",
        derivative_order=1,
        derivative_wrt="time",
        description="Joint velocity trajectory.",
    ),
    "joint_accelerations": QuantityAlias(
        name="joint_accelerations",
        source="trajectory",
        var="trajectory_params",
        derivative_order=2,
        derivative_wrt="time",
        description="Joint acceleration trajectory.",
    ),
    "joint_torques": QuantityAlias(
        name="joint_torques",
        source="state_traj",
        var="trajectory_params",
        owner_type="total_joint",
        owner_name="robot",
        dtype="dynamics",
        field="torque",
        description="Joint torque trajectory computed by the robotics backend.",
    ),
}


def resolve_quantity(
    raw: Any,
    *,
    overrides: Mapping[str, Any] | None = None,
    aliases: Mapping[str, QuantityAlias] = DEFAULT_QUANTITY_ALIASES,
) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        raw_dict = dict(raw)
        if "name" not in raw_dict:
            raise ValueError("quantity object must define name.")
        name = str(raw_dict.pop("name"))
        merged_overrides = dict(raw_dict)
        if overrides is not None:
            merged_overrides.update(dict(overrides))
    else:
        name = str(raw)
        merged_overrides = {} if overrides is None else dict(overrides)

    if name == "":
        raise ValueError("quantity name must be non-empty.")
    alias = aliases.get(name)
    if alias is None:
        raise ValueError(f"unknown quantity {name!r}.")
    if alias.source == "trajectory":
        return _resolve_trajectory_quantity(name, alias, merged_overrides)
    if alias.source == "state_traj":
        return _resolve_state_traj_quantity(name, alias, merged_overrides)
    raise ValueError(f"quantity {name!r} uses unsupported source {alias.source!r}.")


def _resolve_trajectory_quantity(
    name: str,
    alias: QuantityAlias,
    merged_overrides: dict[str, Any],
) -> dict[str, Any]:
    traj: dict[str, Any] = {}
    expr_name = merged_overrides.pop("name", None)
    if expr_name is not None:
        traj["name"] = str(expr_name)
    var = merged_overrides.pop("var", alias.var)
    if var is not None:
        traj["var"] = str(var)
    derivative_order = merged_overrides.pop("derivative", merged_overrides.pop("derivative_order", alias.derivative_order))
    if derivative_order is not None:
        traj["derivative"] = int(derivative_order)
    derivative_wrt = merged_overrides.pop("derivative_wrt", alias.derivative_wrt)
    if derivative_wrt is not None:
        traj["derivative_wrt"] = str(derivative_wrt)
    for src in ("at", "k"):
        if src in merged_overrides:
            traj[src] = deepcopy(merged_overrides.pop(src))
    if merged_overrides:
        keys = ", ".join(sorted(str(k) for k in merged_overrides))
        raise ValueError(f"quantity {name!r} has unsupported override key(s): {keys}.")
    return {"traj": traj}


def _resolve_state_traj_quantity(
    name: str,
    alias: QuantityAlias,
    merged_overrides: dict[str, Any],
) -> dict[str, Any]:
    expr_name = merged_overrides.pop("name", name)
    var = merged_overrides.pop("var", alias.var)
    key: dict[str, Any] = {
        "owner_type": merged_overrides.pop("owner_type", alias.owner_type),
        "owner_name": merged_overrides.pop("owner_name", alias.owner_name),
        "dtype": merged_overrides.pop("dtype", alias.dtype),
        "field": merged_overrides.pop("field", alias.field),
    }
    for optional in ("frame", "rel_frame"):
        value = merged_overrides.pop(optional, None)
        if value is not None:
            key[optional] = deepcopy(value)

    missing = [k for k, v in key.items() if v is None]
    if missing:
        keys = ", ".join(missing)
        raise ValueError(f"quantity {name!r} is missing state key field(s): {keys}.")

    range_dsl: dict[str, Any] = {
        "k0": deepcopy(merged_overrides.pop("k0", 0)),
        "k1": deepcopy(merged_overrides.pop("k1", "last")),
    }
    if "stride" in merged_overrides:
        range_dsl["stride"] = deepcopy(merged_overrides.pop("stride"))
    if "at" in merged_overrides or "k" in merged_overrides:
        k = deepcopy(merged_overrides.pop("at", merged_overrides.pop("k", None)))
        range_dsl = {"k0": k, "k1": k}

    if merged_overrides:
        keys = ", ".join(sorted(str(k) for k in merged_overrides))
        raise ValueError(f"quantity {name!r} has unsupported override key(s): {keys}.")

    inner: dict[str, Any] = {
        "state": key,
    }
    if var is not None:
        inner["var"] = str(var)
    return {
        "op": "stack",
        "name": str(expr_name),
        "range": range_dsl,
        "inner": inner,
    }


def resolve_opt_vals(
    raw: Any,
    *,
    aliases: Mapping[str, OptValAlias] = DEFAULT_OPT_VAL_ALIASES,
) -> OptValResolution:
    if raw is None:
        return OptValResolution(variables=(), aliases={})
    if not isinstance(raw, Mapping):
        raise ValueError("spec.opt_vals must be an object.")

    variables: list[dict[str, Any]] = []
    resolved_aliases: dict[str, str] = {}
    canonical_seen: dict[str, str] = {}
    for alias_raw, value in raw.items():
        alias_name = str(alias_raw)
        if alias_name == "":
            raise ValueError("spec.opt_vals keys must be non-empty.")
        alias = aliases.get(
            alias_name,
            OptValAlias(name=alias_name, canonical_name=alias_name),
        )
        canonical_name = str(alias.canonical_name)
        if canonical_name == "":
            raise ValueError(f"spec.opt_vals.{alias_name}: canonical variable name must be non-empty.")
        previous_alias = canonical_seen.get(canonical_name)
        if previous_alias is not None:
            raise ValueError(
                "spec.opt_vals defines multiple aliases for canonical variable "
                f"{canonical_name!r}: {previous_alias!r} and {alias_name!r}."
            )
        canonical_seen[canonical_name] = alias_name
        resolved_aliases[alias_name] = canonical_name

        entry: dict[str, Any] = {"name": canonical_name}
        if isinstance(value, Mapping):
            entry.update(deepcopy(dict(value)))
            explicit_name = entry.pop("name", entry.pop("var", None))
            if explicit_name is not None:
                entry["name"] = str(explicit_name)
                resolved_aliases[alias_name] = str(explicit_name)
        elif isinstance(value, int):
            entry["dim"] = int(value)
        else:
            entry["init"] = deepcopy(value)
        variables.append(entry)
    return OptValResolution(variables=tuple(variables), aliases=resolved_aliases)


__all__ = [
    "DEFAULT_QUANTITY_ALIASES",
    "DEFAULT_OPT_VAL_ALIASES",
    "OptValAlias",
    "OptValResolution",
    "QuantityAlias",
    "resolve_opt_vals",
    "resolve_quantity",
]
