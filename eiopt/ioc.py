from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .core.state_cache import StateKey
from .model.runtime import LinearizedTerm, ProblemRuntime, StackedTermSlice

Array = np.ndarray


def _project_to_simplex(v: Array) -> Array:
    x = np.asarray(v, dtype=float).reshape(-1)
    n = int(x.size)
    if n == 0:
        return np.zeros((0,), dtype=float)

    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1, dtype=float)
    cond = u - cssv / ind > 0.0
    if not np.any(cond):
        return np.full((n,), 1.0 / n, dtype=float)
    rho = int(np.nonzero(cond)[0][-1])
    theta = float(cssv[rho] / (rho + 1))
    return np.maximum(x - theta, 0.0)


def build_term_gradient_matrix_from_terms(
    term_linearizations: Iterable[LinearizedTerm],
    *,
    n_total: int | None = None,
) -> tuple[Array, list[int]]:
    """Build IOC matrix A whose columns are `J_i.T @ r_i`."""

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
    layout: Iterable[StackedTermSlice],
    *,
    n_total: int | None = None,
) -> tuple[Array, list[int]]:
    """Build IOC matrix A from stacked residual/J and per-term row slices."""

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


def build_term_gradient_matrix(
    runtime: ProblemRuntime,
    *,
    required: Iterable[StateKey] | None = None,
    weighted: bool = False,
    term_indices: Iterable[int] | None = None,
) -> tuple[Array, list[int]]:
    """Linearize selected terms and return IOC matrix A."""

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


def estimate_weights_simplex(
    A: Array | Any,
    *,
    max_iters: int = 2000,
    tol: float = 1e-10,
    step_size: float | None = None,
    x0: Array | Any = None,
    return_info: bool = False,
) -> Array | tuple[Array, dict[str, Any]]:
    """Estimate nonnegative weights on simplex by minimizing `||A @ w||^2`."""

    A_mat = np.asarray(A, dtype=float)
    if A_mat.ndim != 2:
        raise ValueError(f"estimate_weights_simplex: A must be 2D, got shape {A_mat.shape}.")

    _m, n = A_mat.shape
    n = int(n)
    if n <= 0:
        raise ValueError("estimate_weights_simplex: A must have at least one column.")

    if max_iters <= 0:
        raise ValueError(f"estimate_weights_simplex: max_iters must be > 0, got {max_iters}.")
    if tol <= 0.0:
        raise ValueError(f"estimate_weights_simplex: tol must be > 0, got {tol}.")

    if n == 1:
        w_single = np.array([1.0], dtype=float)
        if not return_info:
            return w_single
        r = A_mat @ w_single
        info = {
            "converged": True,
            "iterations": 0,
            "objective": float(0.5 * (r @ r)),
            "residual_norm": float(np.linalg.norm(r)),
            "grad_norm": float(np.linalg.norm(A_mat.T @ r)),
            "step_size": 0.0,
        }
        return w_single, info

    if step_size is None:
        svals = np.linalg.svd(A_mat, compute_uv=False)
        lipschitz = float((svals[0] ** 2) if svals.size > 0 else 1.0)
        lipschitz = max(lipschitz, 1e-12)
        step = 1.0 / lipschitz
    else:
        step = float(step_size)
        if step <= 0.0:
            raise ValueError(f"estimate_weights_simplex: step_size must be > 0, got {step}.")

    if x0 is None:
        w = np.full((n,), 1.0 / n, dtype=float)
    else:
        w = _project_to_simplex(np.asarray(x0, dtype=float).reshape(-1))
        if w.size != n:
            raise ValueError(f"estimate_weights_simplex: x0 size mismatch. Expected {n}, got {w.size}.")

    converged = False
    iterations = int(max_iters)
    for it in range(int(max_iters)):
        Aw = A_mat @ w
        grad = A_mat.T @ Aw
        w_next = _project_to_simplex(w - step * grad)
        if float(np.linalg.norm(w_next - w)) <= tol:
            w = w_next
            converged = True
            iterations = it + 1
            break
        w = w_next

    residual = A_mat @ w
    objective = float(0.5 * (residual @ residual))
    grad = A_mat.T @ residual

    if not return_info:
        return w

    info = {
        "converged": converged,
        "iterations": int(iterations),
        "objective": objective,
        "residual_norm": float(np.linalg.norm(residual)),
        "grad_norm": float(np.linalg.norm(grad)),
        "step_size": float(step),
    }
    return w, info
