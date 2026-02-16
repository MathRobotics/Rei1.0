from __future__ import annotations

from typing import Callable

import numpy as np

Array = np.ndarray


def nls(
    residual: Callable[[Array], Array],
    jacobian: Callable[[Array], Array],
    *,
    x0: Array,
    max_iters: int = 200,
    tol_r: float = 1e-10,
    tol_dx: float = 1e-10,
) -> tuple[Array, float, int, float, float, bool]:
    """Solve a nonlinear least-squares problem with Gauss-Newton.

    Returns:
      (x_star, cost, iters, rnorm, dxnorm, converged)
    """

    x = np.asarray(x0, dtype=float).reshape(-1).copy()

    rnorm = float("inf")
    dxnorm = float("inf")
    converged = False

    for k in range(int(max_iters)):
        r = np.asarray(residual(x), dtype=float).reshape(-1)
        J = np.asarray(jacobian(x), dtype=float)

        rnorm = float(np.linalg.norm(r))
        if rnorm < tol_r:
            converged = True
            return x, float(r @ r), k, rnorm, 0.0, converged

        lhs = J.T @ J
        rhs = -J.T @ r
        dx, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        dx = np.asarray(dx, dtype=float).reshape(-1)
        dxnorm = float(np.linalg.norm(dx))

        if dxnorm < tol_dx:
            converged = True
            return x, float(r @ r), k, rnorm, dxnorm, converged

        x = x + dx

    r = np.asarray(residual(x), dtype=float).reshape(-1)
    rnorm = float(np.linalg.norm(r))
    return x, float(r @ r), int(max_iters), rnorm, dxnorm, converged


__all__ = [
    "nls",
]
