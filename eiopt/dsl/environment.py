from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.time_grid import TimeGrid
from ..expr.expr_register import ExprRegister
from ..model.term import VariablePack


@dataclass
class DslBuildEnv:
    pack: VariablePack
    time: TimeGrid
    expr_register: ExprRegister

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
