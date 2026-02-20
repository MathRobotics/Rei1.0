from __future__ import annotations

from typing import Callable

import numpy as np

from ...core.outcome import SolveOutcome, SolveStats
from ...core.timing import Profiler, ensure_profiler

Array = np.ndarray


def nls(
    residual: Callable[[Array], Array],
    jacobian: Callable[[Array], Array],
    *,
    x0: Array,
    max_iters: int = 200,
    tol_r: float = 1e-10,
    tol_dx: float = 1e-10,
    profiler: Profiler | None = None,
) -> SolveOutcome:
    """Solve a nonlinear least-squares problem with Gauss-Newton.

    Returns:
      SolveOutcome(solution, stats, timing, meta)
    """

    prof = ensure_profiler(profiler)
    with prof.span("solve.setup"):
        x = np.asarray(x0, dtype=float).reshape(-1).copy()
        r0 = np.asarray(residual(x), dtype=float).reshape(-1)
        initial_cost = float(r0 @ r0)

    rnorm = float("inf")
    dxnorm = float("inf")

    def _outcome(
        *,
        status: str,
        iterations: int,
        cost: float,
        rnorm_local: float,
        dxnorm_local: float,
    ) -> SolveOutcome:
        return SolveOutcome(
            solution=x.copy(),
            stats=SolveStats(
                status=str(status),
                iterations=int(iterations),
                initial_objective=float(initial_cost),
                objective=float(cost),
                residual_norm=float(rnorm_local),
                step_norm=float(dxnorm_local),
            ),
            timing=prof.snapshot(),
            meta={"solver": "nls"},
        )

    for k in range(int(max_iters)):
        with prof.span("solve.iter.linearize"):
            r = np.asarray(residual(x), dtype=float).reshape(-1)
            J = np.asarray(jacobian(x), dtype=float)

        rnorm = float(np.linalg.norm(r))
        if rnorm < tol_r:
            return _outcome(
                status="converged",
                iterations=k,
                cost=float(r @ r),
                rnorm_local=rnorm,
                dxnorm_local=0.0,
            )

        with prof.span("solve.iter.step"):
            lhs = J.T @ J
            rhs = -J.T @ r
            dx, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            dx = np.asarray(dx, dtype=float).reshape(-1)
            dxnorm = float(np.linalg.norm(dx))

        if dxnorm < tol_dx:
            return _outcome(
                status="converged",
                iterations=k,
                cost=float(r @ r),
                rnorm_local=rnorm,
                dxnorm_local=dxnorm,
            )

        with prof.span("solve.iter.update"):
            x = x + dx

    with prof.span("solve.finalize"):
        r = np.asarray(residual(x), dtype=float).reshape(-1)
    rnorm = float(np.linalg.norm(r))
    return _outcome(
        status="max_iters",
        iterations=int(max_iters),
        cost=float(r @ r),
        rnorm_local=rnorm,
        dxnorm_local=dxnorm,
    )


__all__ = [
    "nls",
]
