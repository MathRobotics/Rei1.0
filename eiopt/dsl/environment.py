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

    def build_expr(self, spec: dict[str, Any]) -> Any:
        typ = str(spec["type"])
        builder = self.registry.expr.get(typ, None)
        if builder is None:
            raise ValueError(f"unknown expr type: {typ}")
        return builder(self, spec)

    def build_cost(self, spec: dict[str, Any]) -> Any:
        typ = str(spec.get("type", "l2"))
        builder = self.registry.cost.get(typ, None)
        if builder is None:
            raise ValueError(f"unknown cost type: {typ}")
        return builder(spec)
