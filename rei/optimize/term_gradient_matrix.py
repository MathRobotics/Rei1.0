from __future__ import annotations

from collections.abc import Iterable

from ..core.state_cache import StateKey
from ..term_gradient_matrix import (
    Array,
    build_term_gradient_matrix_from_stacked,
    build_term_gradient_matrix_from_terms,
)
from .runtime import NLSRuntime

def build_term_gradient_matrix(
    runtime: NLSRuntime,
    *,
    required: Iterable[StateKey] | None = None,
    weighted: bool = False,
    term_indices: Iterable[int] | None = None,
) -> tuple[Array, list[int]]:
    """Linearize selected terms and return matrix A."""

    linearize_stacked = getattr(runtime, "linearize_stacked_terms_with_layout", None)
    if callable(linearize_stacked):
        r_all, J_all, layout = linearize_stacked(
            required=required,
            weighted=weighted,
            term_indices=term_indices,
        )
        return build_term_gradient_matrix_from_stacked(
            r_all,
            J_all,
            layout,
            n_total=int(runtime.pack.n_total),
        )

    terms = runtime.linearize_terms(
        required=required,
        weighted=weighted,
        term_indices=term_indices,
    )
    return build_term_gradient_matrix_from_terms(terms, n_total=int(runtime.pack.n_total))


__all__ = [
    "build_term_gradient_matrix_from_terms",
    "build_term_gradient_matrix_from_stacked",
    "build_term_gradient_matrix",
]
