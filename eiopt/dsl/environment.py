from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..core.time_grid import TimeGrid
from ..core.trajectory import TrajectoryMap
from ..expr.expr_register import ExprRegister
from ..model.term import VariablePack
from .trajectory import build_trajectory_map, default_steps_from_time


@dataclass
class DslBuildEnv:
    pack: VariablePack
    time: TimeGrid
    expr_register: ExprRegister
    root_dsl: dict[str, Any] | None = None
    trajectory_cache: dict[tuple[int, int | None, int | None], TrajectoryMap] = field(default_factory=dict)

    def build_expr(self, dsl: dict[str, Any]) -> Any:
        typ = str(dsl["type"])
        builder = self.expr_register.expr.get(typ, None)
        if builder is None:
            raise ValueError(f"unknown expr type: {typ}")
        return builder(self, dsl)

    def build_cost(self, dsl: dict[str, Any]) -> Any:
        typ = str(dsl.get("type", "l2"))
        builder = self.expr_register.cost.get(typ, None)
        if builder is None:
            raise ValueError(f"unknown cost type: {typ}")
        return builder(dsl)

    def resolve_trajectory_map(
        self,
        traj_dsl: Mapping[str, Any],
        *,
        default_q_dim: int | None = None,
    ) -> TrajectoryMap:
        default_steps = default_steps_from_time(self.time)
        key = (id(traj_dsl), default_steps, default_q_dim)
        cached = self.trajectory_cache.get(key, None)
        if cached is not None:
            return cached

        traj = build_trajectory_map(
            traj_dsl,
            default_steps=default_steps,
            default_q_dim=default_q_dim,
        )
        self.trajectory_cache[key] = traj
        return traj
