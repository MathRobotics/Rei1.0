from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..core.state_cache import StateKey

Array = np.ndarray


@runtime_checkable
class ProblemPoint(Protocol):
    """Minimal mutable point/state capability used by iterative solvers."""

    @property
    def n_total(self) -> int: ...

    def get_point(self) -> Array: ...

    def set_point(self, x: Array | Any) -> None: ...

    def required_list(self, required: Iterable[StateKey] | None = None) -> list[StateKey]: ...


@runtime_checkable
class EvaluateProblem(ProblemPoint, Protocol):
    """Value evaluation capability y = eval(state)."""

    def eval(self, *, required: Iterable[StateKey] | None = None) -> Array: ...


@runtime_checkable
class LinearizedProblem(EvaluateProblem, Protocol):
    """Linearization capability y, J = linearize(state)."""

    def linearize(self, *, required: Iterable[StateKey] | None = None) -> tuple[Array, Array]: ...


@runtime_checkable
class OperatorProblem(ProblemPoint, Protocol):
    """Operator capability via Jacobian-vector products."""

    def jvp(self, v: Array | Any, *, required: Iterable[StateKey] | None = None) -> Array: ...

    def vjp(self, w: Array | Any, *, required: Iterable[StateKey] | None = None) -> Array: ...


@runtime_checkable
class ProjectProblem(ProblemPoint, Protocol):
    """Projection capability for constrained/manifold dynamics."""

    def project(self, x: Array | Any) -> Array: ...


@runtime_checkable
class ConstraintProblem(ProblemPoint, Protocol):
    """Explicit constraint residual/Jacobian capability."""

    def constraint(self, *, required: Iterable[StateKey] | None = None) -> Array: ...

    def jacobian_constraint(self, *, required: Iterable[StateKey] | None = None) -> Array: ...


__all__ = [
    "Array",
    "ProblemPoint",
    "EvaluateProblem",
    "LinearizedProblem",
    "OperatorProblem",
    "ProjectProblem",
    "ConstraintProblem",
]
