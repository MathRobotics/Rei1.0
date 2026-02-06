from __future__ import annotations

from typing import Any, Optional, Callable, Iterable

import numpy as np

from ..core.state_cache import StateKey
from ..model.problem import Problem
from ..model.term import VariablePack


def solve_gauss_newton(
    problem: Problem,
    variables: VariablePack,
    max_iters: int = 20,
    *,
    ctx: Any = None,
    required: Optional[Iterable[StateKey]] = None,
    tol_r: float = 1e-10,
    tol_dx: float = 1e-12,
    on_iter: Optional[Callable[[int, float, float], None]] = None,
) -> None:
    """Minimal Gauss-Newton loop for Expr-based Problem."""

    for k in range(max_iters):
        if ctx is not None and hasattr(ctx, "state") and ctx.state is not None:
            if hasattr(ctx.state, "update_if_needed"):
                ctx.state.update_if_needed(variables, time=getattr(ctx, "time", None), required=required)

        r_all, J_all = problem.linearize(ctx=ctx, time=getattr(ctx, "time", None), required=required)

        nr = float(np.linalg.norm(r_all))
        if on_iter is not None:
            on_iter(k, nr, 0.0)

        if nr < tol_r:
            break

        lhs = J_all.T @ J_all
        rhs = -J_all.T @ r_all

        dx, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        ndx = float(np.linalg.norm(dx))

        if on_iter is not None:
            on_iter(k, nr, ndx)

        if ndx < tol_dx:
            break

        variables.apply_dx(dx)
        variables.revision += 1
