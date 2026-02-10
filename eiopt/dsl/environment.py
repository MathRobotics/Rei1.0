from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.time_grid import TimeGrid
from ..expr.registry import Registry
from ..model.term import VariablePack


@dataclass
class DslBuildEnv:
    pack: VariablePack
    time: TimeGrid
    registry: Registry

    def build_expr(self, dsl: dict[str, Any]) -> Any:
        typ = str(dsl["type"])
        builder = self.registry.expr.get(typ, None)
        if builder is None:
            raise ValueError(f"unknown expr type: {typ}")
        return builder(self, dsl)

    def build_cost(self, dsl: dict[str, Any]) -> Any:
        typ = str(dsl.get("type", "l2"))
        builder = self.registry.cost.get(typ, None)
        if builder is None:
            raise ValueError(f"unknown cost type: {typ}")
        return builder(dsl)
