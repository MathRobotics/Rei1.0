from __future__ import annotations

from typing import Any

import numpy as np

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


def estimate_weights_simplex(
    A: Array | Any,
    *,
    max_iters: int = 2000,
    tol: float = 1e-10,
    step_size: float | None = None,
    x0: Array | Any = None,
    return_info: bool = False,
) -> Array | tuple[Array, dict[str, Any]]:
    """Estimate nonnegative simplex weights by minimizing `||A @ w||^2`."""

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
