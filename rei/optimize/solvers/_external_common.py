"""Shared objective evaluation and setup for optional solver integrations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from ...core.outcome import SolveOutcome
from ...core.state_cache import StateKey
from ...problem import LinearizedProblem
from ...xops import as_vec

Array = np.ndarray
SolveResult = SolveOutcome
IterCallback = Callable[[int, float, float], None]


def _format_option_names(keys: Iterable[str]) -> str:
    return ", ".join(sorted(str(k) for k in keys))


class _LinearizedObjective:
    def __init__(
        self,
        problem: LinearizedProblem,
        *,
        required: Iterable[StateKey] | None = None,
    ) -> None:
        self.problem = problem
        self.required = None if required is None else list(required)
        self._last_x: Array | None = None
        self._last_cost: float = float("nan")
        self._last_grad: Array | None = None
        self._last_rnorm: float = float("inf")

    def eval(self, x: Array) -> tuple[float, Array, float]:
        x_vec = as_vec(x, expected_size=int(self.problem.n_total), name="x")
        if self._last_x is not None and np.array_equal(self._last_x, x_vec):
            if self._last_grad is None:
                raise RuntimeError("internal error: gradient cache is missing.")
            return self._last_cost, self._last_grad, self._last_rnorm

        self.problem.set_point(x_vec)
        r_all, J_all = self.problem.linearize(required=self.required)
        r = np.asarray(r_all, dtype=float).reshape(-1)
        J = np.asarray(J_all, dtype=float)

        cost = float(r @ r)
        grad = np.asarray(2.0 * (J.T @ r), dtype=float).reshape(-1)
        rnorm = float(np.linalg.norm(r))

        self._last_x = x_vec.copy()
        self._last_cost = cost
        self._last_grad = grad
        self._last_rnorm = rnorm
        return cost, grad, rnorm


def _extract_x_from_callback_arg(xk: Any, *, size: int) -> Array | None:
    try:
        return as_vec(xk, expected_size=size, name="callback.x")
    except Exception:
        pass

    x_attr = getattr(xk, "x", None)
    if x_attr is None:
        return None
    try:
        return as_vec(x_attr, expected_size=size, name="callback.x")
    except Exception:
        return None


def _initialize_linear_problem_point(
    linear_problem: LinearizedProblem,
    *,
    x0: Array | Any = None,
) -> Array:
    n_total = int(linear_problem.n_total)
    if x0 is not None:
        linear_problem.set_point(as_vec(x0, expected_size=n_total, name="x0"))
    return np.asarray(linear_problem.get_point(), dtype=float).reshape(-1).copy()

