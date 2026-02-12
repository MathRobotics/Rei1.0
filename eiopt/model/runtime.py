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

    def _term_display_name(self, index: int) -> str:
        expr, _cost = self.problem.terms[index]
        name = getattr(expr, "name", None)
        if isinstance(name, str) and name:
            return name
        return expr.__class__.__name__

    def _resolve_term_index(self, term: int | str) -> int:
        if isinstance(term, int):
            idx = int(term)
            if idx < 0 or idx >= len(self.problem.terms):
                raise IndexError(
                    f"set_cost_weight: term index out of range: {idx}. "
                    f"Expected 0..{len(self.problem.terms) - 1}."
                )
            return idx

        term_name = str(term)
        if term_name == "":
            raise ValueError("set_cost_weight: term name must be non-empty.")

        matches = [
            i
            for i, (expr, _cost) in enumerate(self.problem.terms)
            if getattr(expr, "name", None) == term_name
        ]
        if len(matches) == 0:
            raise ValueError(f"set_cost_weight: no term found with name={term_name!r}.")
        if len(matches) > 1:
            idxs = ", ".join(str(i) for i in matches)
            raise ValueError(
                f"set_cost_weight: multiple terms matched name={term_name!r} at indices [{idxs}]. "
                "Use an explicit term index."
            )
        return matches[0]

    def set_cost_weight(self, term: int | str, w: Any) -> int:
        """Update one term's cost weight and invalidate linearization cache.

        `term` can be either the term index or the term expression name.
        The target cost must implement `set_weight(w)`.
        """

        idx = self._resolve_term_index(term)
        _expr, cost = self.problem.terms[idx]
        set_weight = getattr(cost, "set_weight", None)
        if not callable(set_weight):
            raise TypeError(
                f"set_cost_weight: term[{idx}] '{self._term_display_name(idx)}' "
                f"uses cost type '{type(cost).__name__}' which does not support runtime weight updates."
            )

        set_weight(w)
        self.problem.invalidate_cache()
        return idx
