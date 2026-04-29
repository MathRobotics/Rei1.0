from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.state_cache import StateKey
from ..xops import as_vec, set_runtime_x
from .caps import Array, ConstraintProblem, LinearizedProblem
from .runtime_helpers import runtime_n_total, runtime_point, runtime_required_list


@dataclass
class NLSRuntimeLinearProblem:
    """Adapt an NLSRuntime-like object to a generic LinearizedProblem."""

    runtime: Any
    weighted: bool = True
    term_indices: Sequence[int] | None = None

    @property
    def n_total(self) -> int:
        return runtime_n_total(self.runtime, adapter_name="NLSRuntimeLinearProblem")

    def get_point(self) -> Array:
        return runtime_point(self.runtime, adapter_name="NLSRuntimeLinearProblem")

    def set_point(self, x: Array | Any) -> None:
        x_vec = as_vec(x, expected_size=int(self.n_total), name="x")
        set_runtime_x(self.runtime, x_vec, name="x")

    def required_list(self, required: Iterable[StateKey] | None = None) -> list[StateKey]:
        return runtime_required_list(self.runtime, required)

    def linearize(self, *, required: Iterable[StateKey] | None = None) -> tuple[Array, Array]:
        linearize_stacked = getattr(self.runtime, "linearize_stacked_terms", None)
        if callable(linearize_stacked):
            return linearize_stacked(
                required=required,
                weighted=bool(self.weighted),
                term_indices=self.term_indices,
            )

        linearize = getattr(self.runtime, "linearize", None)
        if not callable(linearize):
            raise AttributeError(
                "NLSRuntimeLinearProblem: runtime must expose linearize_stacked_terms(...) "
                "or linearize(...)."
            )
        r, J = linearize(required=required)
        return np.asarray(r, dtype=float).reshape(-1), np.asarray(J, dtype=float)

    def eval(self, *, required: Iterable[StateKey] | None = None) -> Array:
        r, _J = self.linearize(required=required)
        return np.asarray(r, dtype=float).reshape(-1)

    def jvp(self, v: Array | Any, *, required: Iterable[StateKey] | None = None) -> Array:
        _r, J = self.linearize(required=required)
        v_vec = as_vec(v, expected_size=int(self.n_total), name="v")
        return np.asarray(J, dtype=float) @ v_vec

    def vjp(self, w: Array | Any, *, required: Iterable[StateKey] | None = None) -> Array:
        r, J = self.linearize(required=required)
        w_vec = as_vec(w, expected_size=int(np.asarray(r, dtype=float).reshape(-1).size), name="w")
        return np.asarray(J, dtype=float).T @ w_vec


@dataclass
class NLSRuntimeConstraintProblem:
    """Adapt NLSRuntime-like equality/inequality terms as constraint capability."""

    runtime: Any
    kind: str | None = None
    weighted: bool = False

    @property
    def n_total(self) -> int:
        return runtime_n_total(self.runtime, adapter_name="NLSRuntimeConstraintProblem")

    def get_point(self) -> Array:
        return runtime_point(self.runtime, adapter_name="NLSRuntimeConstraintProblem")

    def set_point(self, x: Array | Any) -> None:
        x_vec = as_vec(x, expected_size=int(self.n_total), name="x")
        set_runtime_x(self.runtime, x_vec, name="x")

    def required_list(self, required: Iterable[StateKey] | None = None) -> list[StateKey]:
        return runtime_required_list(self.runtime, required)

    def _linearized_terms(self, *, required: Iterable[StateKey] | None = None) -> list[Any]:
        linearize_constraint_terms = getattr(self.runtime, "linearize_constraint_terms", None)
        if not callable(linearize_constraint_terms):
            raise AttributeError(
                "NLSRuntimeConstraintProblem: runtime must expose linearize_constraint_terms(...)."
            )
        return list(
            linearize_constraint_terms(
                required=required,
                kind=self.kind,
                weighted=bool(self.weighted),
            )
        )

    def constraint(self, *, required: Iterable[StateKey] | None = None) -> Array:
        terms = self._linearized_terms(required=required)
        if len(terms) == 0:
            return np.zeros((0,), dtype=float)
        return np.concatenate([np.asarray(t.residual, dtype=float).reshape(-1) for t in terms], axis=0)

    def jacobian_constraint(self, *, required: Iterable[StateKey] | None = None) -> Array:
        terms = self._linearized_terms(required=required)
        if len(terms) == 0:
            return np.zeros((0, int(self.n_total)), dtype=float)
        return np.vstack([np.asarray(t.jacobian, dtype=float) for t in terms])

    def eval(self, *, required: Iterable[StateKey] | None = None) -> Array:
        return self.constraint(required=required)

    def linearize(self, *, required: Iterable[StateKey] | None = None) -> tuple[Array, Array]:
        return self.constraint(required=required), self.jacobian_constraint(required=required)


def as_linearized_problem(
    problem: Any,
    *,
    weighted: bool = True,
    term_indices: Sequence[int] | None = None,
) -> LinearizedProblem:
    """Coerce input to LinearizedProblem while preserving existing runtime support."""

    if isinstance(problem, LinearizedProblem):
        return problem

    if hasattr(problem, "linearize_stacked_terms") and hasattr(problem, "pack"):
        return NLSRuntimeLinearProblem(
            runtime=problem,
            weighted=bool(weighted),
            term_indices=term_indices,
        )

    missing = []
    for name in ("n_total", "get_point", "set_point", "required_list", "linearize"):
        if not hasattr(problem, name):
            missing.append(name)
    if missing:
        miss = ", ".join(missing)
        raise TypeError(
            "as_linearized_problem: object is not a supported linearized problem. "
            f"Missing attribute(s): {miss}."
        )
    return problem


__all__ = [
    "NLSRuntimeLinearProblem",
    "NLSRuntimeConstraintProblem",
    "as_linearized_problem",
]
