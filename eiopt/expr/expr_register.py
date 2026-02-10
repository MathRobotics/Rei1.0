from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class ExprRegister:
    expr: dict[str, Callable[[Any, dict], Any]]
    cost: dict[str, Callable[[dict], Any]]

    def __init__(self) -> None:
        self.expr = {}
        self.cost = {}

    def register_expr(self, typ: str, fn: Callable[[Any, dict], Any]) -> None:
        self.expr[typ] = fn

    def register_cost(self, typ: str, fn: Callable[[dict], Any]) -> None:
        self.cost[typ] = fn
