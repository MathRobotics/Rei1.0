from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.state_cache import StateKey
from ..problem import ConstraintProblem, NLSRuntimeConstraintProblem, ProjectProblem
from ..problem.runtime_helpers import runtime_n_total, runtime_point, runtime_required_list
from ..xops import as_vec, set_runtime_x

Array = np.ndarray


@dataclass
class RuntimeIdentityProjector:
    """Add identity projection capability to runtime-like objects with a pack."""

    runtime: Any

    @property
    def n_total(self) -> int:
        return runtime_n_total(self.runtime, adapter_name="RuntimeIdentityProjector")

    def get_point(self) -> Array:
        return runtime_point(self.runtime, adapter_name="RuntimeIdentityProjector")

    def set_point(self, x: Array | Any) -> None:
        x_vec = as_vec(x, expected_size=int(self.n_total), name="x")
        set_runtime_x(self.runtime, x_vec, name="x")

    def required_list(self, required: Iterable[StateKey] | None = None) -> list[StateKey]:
        return runtime_required_list(self.runtime, required)

    def project(self, x: Array | Any) -> Array:
        return as_vec(x, expected_size=int(self.n_total), name="x").copy()


def as_constraint_problem(
    problem: Any,
    *,
    kind: str | None = None,
    weighted: bool = False,
) -> ConstraintProblem:
    """Coerce input to an explicit constraint capability."""

    if isinstance(problem, ConstraintProblem):
        return problem

    if hasattr(problem, "linearize_constraint_terms") and hasattr(problem, "pack"):
        return NLSRuntimeConstraintProblem(
            runtime=problem,
            kind=kind,
            weighted=bool(weighted),
        )

    missing: list[str] = []
    for name in ("n_total", "get_point", "set_point", "required_list", "constraint", "jacobian_constraint"):
        if not hasattr(problem, name):
            missing.append(name)
    if missing:
        miss = ", ".join(missing)
        raise TypeError(
            "as_constraint_problem: object is not a supported constraint problem. "
            f"Missing attribute(s): {miss}."
        )
    return problem


def as_project_problem(problem: Any) -> ProjectProblem:
    """Coerce input to a projection capability."""

    if isinstance(problem, ProjectProblem):
        return problem

    if hasattr(problem, "pack"):
        return RuntimeIdentityProjector(problem)

    missing: list[str] = []
    for name in ("n_total", "get_point", "set_point", "required_list", "project"):
        if not hasattr(problem, name):
            missing.append(name)
    if missing:
        miss = ", ".join(missing)
        raise TypeError(
            "as_project_problem: object is not a supported project problem. "
            f"Missing attribute(s): {miss}."
        )
    return problem


__all__ = [
    "RuntimeIdentityProjector",
    "as_constraint_problem",
    "as_project_problem",
]
