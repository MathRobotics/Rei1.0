from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np

from ..core.state_cache import StateKey
from ..model.runtime import ProblemRuntime

Array = np.ndarray


def solve_gauss_newton(
    runtime: ProblemRuntime,
    max_iters: int = 20,
    *,
    required: Optional[Iterable[StateKey]] = None,
    tol_r: float = 1e-10,
    tol_dx: float = 1e-12,
    on_iter: Optional[Callable[[int, float, float], None]] = None,
) -> tuple[Array, float, int, float, float, bool]:
    """Minimal Gauss-Newton loop for a `ProblemRuntime`.

    Returns:
      (x_star, cost, iters, rnorm, dxnorm, converged)
    """

    rnorm = float("inf")
    dxnorm = float("inf")
    converged = False

    variables = runtime.pack

    for k in range(int(max_iters)):
        r_all, J_all = runtime.linearize(required=required)
        rnorm = float(np.linalg.norm(r_all))

        if on_iter is not None:
            on_iter(k, rnorm, 0.0)

        if rnorm < tol_r:
            converged = True
            x_star = np.asarray(variables.get(), dtype=float).reshape(-1).copy()
            cost = float(r_all @ r_all)
            return x_star, cost, k, rnorm, 0.0, converged

        lhs = J_all.T @ J_all
        rhs = -J_all.T @ r_all

        dx, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        dx = np.asarray(dx, dtype=float).reshape(-1)
        dxnorm = float(np.linalg.norm(dx))

        if on_iter is not None:
            on_iter(k, rnorm, dxnorm)

        if dxnorm < tol_dx:
            converged = True
            x_star = np.asarray(variables.get(), dtype=float).reshape(-1).copy()
            cost = float(r_all @ r_all)
            return x_star, cost, k, rnorm, dxnorm, converged

        variables.apply_dx(dx)

    # max_iters exhausted: report residual at final x (after last update)
    r_all, _J_all = runtime.linearize(required=required)
    rnorm = float(np.linalg.norm(r_all))
    x_star = np.asarray(variables.get(), dtype=float).reshape(-1).copy()
    cost = float(r_all @ r_all)
    return x_star, cost, int(max_iters), rnorm, dxnorm, converged
