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
    "DEFAULT_OPT_VAL_ALIASES",
    "OptValAlias",
    "OptValResolution",
    "resolve_opt_vals",
]
