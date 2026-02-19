from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..problem import (
    EvaluateProblem,
    LinearizedProblem,
    OperatorProblem,
    ProblemPoint,
    as_linearized_problem,
)

EquationPoint = ProblemPoint
EvaluateEquation = EvaluateProblem
LinearizedEquation = LinearizedProblem
OperatorEquation = OperatorProblem


def as_linear_equation_problem(
    problem: Any,
    *,
    weighted: bool = True,
    term_indices: Sequence[int] | None = None,
) -> LinearizedEquation:
    """Coerce input to a linearized equation problem."""

    return as_linearized_problem(
        problem,
        weighted=weighted,
        term_indices=term_indices,
    )


__all__ = [
    "EquationPoint",
    "EvaluateEquation",
    "LinearizedEquation",
    "OperatorEquation",
    "as_linear_equation_problem",
]
