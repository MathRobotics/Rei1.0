from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.state_cache import StateKey
from ..problem import LinearizedProblem, as_linearized_problem
from ..xops import as_vec

Array = np.ndarray


@dataclass
class SolverProblem:
    """External-solver view of a Rei runtime or linearized problem.

    The adapter exposes residual/Jacobian and scalar objective/gradient callables
    without choosing a solver implementation.
    """

    source: Any
    weighted: bool = True
    term_indices: Sequence[int] | None = None
    required: Iterable[StateKey] | None = None
    linear_problem: LinearizedProblem = field(init=False)
    x0: Array = field(init=False)

    def __post_init__(self) -> None:
        self.linear_problem = as_linearized_problem(
            self.source,
            weighted=bool(self.weighted),
            term_indices=self.term_indices,
        )
        self.x0 = np.asarray(self.linear_problem.get_point(), dtype=float).reshape(-1).copy()
        self.required = None if self.required is None else tuple(self.required)

    @property
    def n_total(self) -> int:
        return int(self.linear_problem.n_total)

    def get_point(self) -> Array:
        return np.asarray(self.linear_problem.get_point(), dtype=float).reshape(-1).copy()

    def set_point(self, x: Array | Any) -> None:
        self.linear_problem.set_point(as_vec(x, expected_size=self.n_total, name="x"))

    def linearize(self, x: Array | Any | None = None) -> tuple[Array, Array]:
        if x is not None:
            self.set_point(x)
        r, J = self.linear_problem.linearize(required=self.required)
        return np.asarray(r, dtype=float).reshape(-1), np.asarray(J, dtype=float)

    def residual(self, x: Array | Any | None = None) -> Array:
        r, _J = self.linearize(x)
        return r

    def jacobian(self, x: Array | Any | None = None) -> Array:
        _r, J = self.linearize(x)
        return J

    def objective(self, x: Array | Any | None = None) -> float:
        r = self.residual(x)
        return float(r @ r)

    def gradient(self, x: Array | Any | None = None) -> Array:
        r, J = self.linearize(x)
        return np.asarray(2.0 * (J.T @ r), dtype=float).reshape(-1)

    def objective_and_gradient(self, x: Array | Any | None = None) -> tuple[float, Array]:
        r, J = self.linearize(x)
        return float(r @ r), np.asarray(2.0 * (J.T @ r), dtype=float).reshape(-1)


def as_solver_problem(
    source: Any,
    *,
    weighted: bool = True,
    term_indices: Sequence[int] | None = None,
    required: Iterable[StateKey] | None = None,
) -> SolverProblem:
    return SolverProblem(
        source=source,
        weighted=bool(weighted),
        term_indices=term_indices,
        required=required,
    )


__all__ = [
    "SolverProblem",
    "as_solver_problem",
]
