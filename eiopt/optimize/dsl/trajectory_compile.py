from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ...core.trajectory import TrajectoryMap
from .dsl_ops import find_var_dsl
from .trajectory import (
    build_trajectory_map,
    build_trajectory_maps_with_derivatives,
    default_dt_from_time,
    default_steps_from_time,
)
from .variable_utils import normalize_variable_dsl


@dataclass(frozen=True)
class PreparedTrajectoryProblemDsl:
    dsl: dict[str, Any]
    p_var: str
    trajectory_map: TrajectoryMap
    trajectory_derivative_maps: dict[int, TrajectoryMap]
    dt: float
    model_order: int


def _mapping_as_dict(mapping: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    if isinstance(mapping, dict):
        return mapping
    try:
        return dict(mapping)
    except Exception as e:
        raise TypeError(f"{where} must be a mapping.") from e


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


def _resolve_p_var_name(*, trajectory_dsl: Mapping[str, Any], p_var: str | None) -> str:
    if p_var is not None:
        name = str(p_var).strip()
    else:
        name = str(trajectory_dsl.get("var", "p")).strip()
    if name == "":
        raise ValueError("trajectory.var must be non-empty.")
    return name


def _ensure_variables_list(dsl: dict[str, Any]) -> list[dict[str, Any]]:
    variables = dsl.get("variables", None)
    if variables is None:
        out: list[dict[str, Any]] = []
        dsl["variables"] = out
        return out
    if not isinstance(variables, list):
        raise ValueError("DSL variables must be a list.")
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(variables):
        if not isinstance(entry, Mapping):
            raise ValueError(f"DSL variables[{i}] must be a mapping.")
        out.append(_mapping_as_dict(entry, where=f"dsl.variables[{i}]"))
    dsl["variables"] = out
    return out


def _ensure_variable_entry(dsl: dict[str, Any], *, name: str) -> dict[str, Any]:
    variables = _ensure_variables_list(dsl)
    var_dsl = find_var_dsl(dsl, name=name)
    if var_dsl is not None:
        return _mapping_as_dict(var_dsl, where=f"dsl.variables[{name!r}]")

    entry = {"name": str(name)}
    variables.append(entry)
    return entry


def _normalize_variable_entry(
    *,
    var_dsl: dict[str, Any],
    expected_dim: int,
    where: str,
) -> None:
    normalized = normalize_variable_dsl(
        var_dsl,
        expected_dim=int(expected_dim),
        where=where,
    )
    var_dsl.clear()
    var_dsl.update(normalized)


def prepare_trajectory_problem_dsl(
    dsl: Mapping[str, Any],
    *,
    p_var: str | None = None,
    model_dof: int | None = None,
    model_order: int = 1,
    max_derivative_order: int | None = None,
    derivative_wrt: str = "time",
    default_steps: int | None = None,
    default_q_dim: int | None = None,
    default_dt: float | None = None,
) -> PreparedTrajectoryProblemDsl:
    """Prepare a normalized trajectory DSL and derivative maps for backend compile paths."""

    dsl_dict = deepcopy(_mapping_as_dict(dsl, where="dsl"))
    trajectory_dsl_raw = dsl_dict.get("trajectory", None)
    if not isinstance(trajectory_dsl_raw, Mapping):
        raise ValueError("DSL must contain [trajectory] section.")
    trajectory_dsl = _mapping_as_dict(trajectory_dsl_raw, where="dsl.trajectory")

    p_var_name = _resolve_p_var_name(trajectory_dsl=trajectory_dsl, p_var=p_var)

    if default_steps is None:
        default_steps = default_steps_from_time(dsl_dict)
    if default_q_dim is None:
        default_q_dim = model_dof

    traj_map = build_trajectory_map(
        trajectory_dsl,
        default_steps=default_steps,
        default_q_dim=default_q_dim,
    )
    trajectory_dsl.setdefault("steps", int(traj_map.steps))
    trajectory_dsl.setdefault("q_dim", int(traj_map.q_dim))
    dt = _resolve_dt(dsl_dict, default_dt=default_dt)

    model_order_i = int(model_order)
    if model_order_i <= 0:
        raise ValueError(f"model_order must be >= 1, got {model_order_i}.")

    if max_derivative_order is None:
        max_derivative_order_use = max(0, model_order_i - 1)
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

    p_var_dsl = _ensure_variable_entry(dsl_dict, name=p_var_name)
    _normalize_variable_entry(
        var_dsl=p_var_dsl,
        expected_dim=int(traj_map.p_dim),
        where=f"variable {p_var_name!r}",
    )

    return PreparedTrajectoryProblemDsl(
        dsl=dsl_dict,
        p_var=p_var_name,
        trajectory_map=traj_map,
        trajectory_derivative_maps=traj_maps_by_order,
        dt=float(dt),
        model_order=model_order_i,
    )
