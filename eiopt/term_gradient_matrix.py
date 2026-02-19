from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np

Array = np.ndarray


class LinearizedTermLike(Protocol):
    term_index: int
    residual: Any
    jacobian: Any


class StackedTermSliceLike(Protocol):
    term_index: int
    row_start: int
    row_stop: int


def build_term_gradient_matrix_from_terms(
    term_linearizations: Iterable[LinearizedTermLike],
    *,
    n_total: int | None = None,
) -> tuple[Array, list[int]]:
    """Build matrix A whose columns are per-term gradients `J_i.T @ r_i`."""

    terms = list(term_linearizations)
    if len(terms) == 0:
        rows = 0 if n_total is None else int(n_total)
        if rows < 0:
            raise ValueError(f"build_term_gradient_matrix_from_terms: n_total must be >= 0, got {rows}.")
        return np.zeros((rows, 0), dtype=float), []

    n_cols = len(terms)
    if n_total is None:
        n_total = int(np.asarray(terms[0].jacobian, dtype=float).shape[1])
    n_total = int(n_total)
    if n_total < 0:
        raise ValueError(f"build_term_gradient_matrix_from_terms: n_total must be >= 0, got {n_total}.")

    A = np.zeros((n_total, n_cols), dtype=float)
    term_indices: list[int] = []
    for j, term in enumerate(terms):
        r = np.asarray(term.residual, dtype=float).reshape(-1)
        J = np.asarray(term.jacobian, dtype=float)
        if J.ndim != 2:
            raise ValueError(
                f"build_term_gradient_matrix_from_terms: jacobian for term[{term.term_index}] must be 2D, got {J.shape}."
            )
        if J.shape[0] != r.size:
            raise ValueError(
                "build_term_gradient_matrix_from_terms: row mismatch for "
                f"term[{term.term_index}]. residual size={r.size}, jacobian shape={J.shape}."
            )
        if J.shape[1] != n_total:
            raise ValueError(
                "build_term_gradient_matrix_from_terms: column mismatch for "
                f"term[{term.term_index}]. Expected {n_total}, got {J.shape[1]}."
            )
        A[:, j] = np.asarray(J.T @ r, dtype=float).reshape(-1)
        term_indices.append(int(term.term_index))
    return A, term_indices


def build_term_gradient_matrix_from_stacked(
    r_all: Array | Any,
    J_all: Array | Any,
    layout: Iterable[StackedTermSliceLike],
    *,
    n_total: int | None = None,
) -> tuple[Array, list[int]]:
    """Build matrix A from stacked residual/J and per-term row slices."""

    r = np.asarray(r_all, dtype=float).reshape(-1)
    J = np.asarray(J_all, dtype=float)
    if J.ndim != 2:
        raise ValueError(
            f"build_term_gradient_matrix_from_stacked: J_all must be 2D, got shape {J.shape}."
        )
    if J.shape[0] != r.size:
        raise ValueError(
            "build_term_gradient_matrix_from_stacked: row mismatch between r_all and J_all. "
            f"len(r_all)={r.size}, J_all.shape={J.shape}."
        )

    if n_total is None:
        n_total = int(J.shape[1])
    n_total = int(n_total)
    if n_total < 0:
        raise ValueError(
            f"build_term_gradient_matrix_from_stacked: n_total must be >= 0, got {n_total}."
        )
    if int(J.shape[1]) != n_total:
        raise ValueError(
            "build_term_gradient_matrix_from_stacked: column mismatch. "
            f"J_all.shape[1]={int(J.shape[1])}, expected n_total={n_total}."
        )

    term_layout = list(layout)
    n_cols = len(term_layout)
    A = np.zeros((n_total, n_cols), dtype=float)
    term_indices: list[int] = []
    n_rows = int(r.size)

    for j, item in enumerate(term_layout):
        start = int(item.row_start)
        stop = int(item.row_stop)
        if start < 0 or stop < start or stop > n_rows:
            raise ValueError(
                "build_term_gradient_matrix_from_stacked: invalid row slice for "
                f"term[{int(item.term_index)}]: [{start}:{stop}] with total rows={n_rows}."
            )
        r_i = r[start:stop]
        J_i = J[start:stop, :]
        A[:, j] = np.asarray(J_i.T @ r_i, dtype=float).reshape(-1)
        term_indices.append(int(item.term_index))
    return A, term_indices


__all__ = [
    "build_term_gradient_matrix_from_terms",
    "build_term_gradient_matrix_from_stacked",
]
