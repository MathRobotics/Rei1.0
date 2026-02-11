from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.state_cache import StateKey
from .problem import Problem
from .term import RuntimeContext, VariablePack

Array = np.ndarray


def _dedupe_required(keys: Iterable[StateKey]) -> list[StateKey]:
    out: list[StateKey] = []
    seen: set[StateKey] = set()
    for k in keys:
        if k in seen:
            continue
        out.append(k)
        seen.add(k)
    return out


def collect_required(problem: Problem) -> list[StateKey]:
    req: list[StateKey] = []
    for expr, _cost in problem.terms:
        deps = getattr(expr, "deps", None)
        if callable(deps):
            req.extend(list(deps()))
    return _dedupe_required(req)


@dataclass
class ProblemRuntime:
    """Runtime holder for a compiled problem.

    This keeps mutable execution concerns (pack/state/time/required) out of `Problem`.
    """

    problem: Problem
    ctx: RuntimeContext
    required: list[StateKey] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.required = _dedupe_required(self.required)

    @classmethod
    def from_problem(
        cls,
        problem: Problem,
        *,
        state: Any = None,
        time: Any = None,
        required: Iterable[StateKey] | None = None,
    ) -> "ProblemRuntime":
        ctx = RuntimeContext(
            pack=problem.variables,
            state=state,
            time=time,
            revision=int(getattr(time, "revision", 0)),
        )
        req = [] if required is None else _dedupe_required(required)
        return cls(problem=problem, ctx=ctx, required=req)

    @property
    def pack(self) -> VariablePack:
        return self.ctx.pack

    @property
    def state(self) -> Any:
        return self.ctx.state

    @property
    def time(self) -> Any:
        return self.ctx.time

    def required_list(self, required: Iterable[StateKey] | None = None) -> list[StateKey]:
        if required is None:
            if not self.required:
                self.required = collect_required(self.problem)
            return list(self.required)
        return _dedupe_required(required)

    def update_state_if_needed(self, *, required: Iterable[StateKey] | None = None) -> None:
        req = self.required_list(required)
        update_if_needed = getattr(self.state, "update_if_needed", None) if self.state is not None else None
        if callable(update_if_needed):
            update_if_needed(self.pack, time=self.time, required=req)

    def linearize(self, *, required: Iterable[StateKey] | None = None) -> tuple[Array, Array]:
        req = self.required_list(required)
        self.update_state_if_needed(required=req)
        return self.problem.linearize(ctx=self.ctx, time=self.time, required=req)

    def cost_value(self, *, required: Iterable[StateKey] | None = None) -> float:
        r_all, _ = self.linearize(required=required)
        return float(r_all @ r_all)
