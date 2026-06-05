from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from ...core.outcome import SolveOutcome, SolveStats
from ...core.state_cache import StateKey
from ...core.timing import Profiler, ensure_profiler
from ...problem import LinearizedProblem, as_linearized_problem
from ...xops import as_vec

Array = np.ndarray

def solve_gauss_newton(
    problem: Any,
    max_iters: int = 20,
    *,
    x0: Array | Any = None,
    required: Iterable[StateKey] | None = None,
    weighted: bool = True,
    term_indices: Iterable[int] | None = None,
    tol_r: float = 1e-10,
    tol_dx: float = 1e-12,
    damping: float = 1e-8,
    line_search: bool = True,
    ls_beta: float = 0.5,
    ls_min_step: float = 1e-8,
    ls_max_iters: int = 12,
    on_iter: Callable[[int, float, float], None] | None = None,
    profiler: Profiler | None = None,
) -> SolveOutcome:
    """Minimal Gauss-Newton loop for a linearized residual problem.

    Returns:
      SolveOutcome with converged/max-iter/stalled status and timing spans.
    """

    prof = ensure_profiler(profiler)
    with prof.span("solve.setup"):
        linear_problem: LinearizedProblem = as_linearized_problem(
            problem,
            weighted=bool(weighted),
            term_indices=None if term_indices is None else tuple(int(i) for i in term_indices),
        )
        n_total = int(linear_problem.n_total)
        if x0 is not None:
            linear_problem.set_point(as_vec(x0, expected_size=n_total, name="x0"))
        x0_start = np.asarray(linear_problem.get_point(), dtype=float).reshape(-1).copy()
        req = linear_problem.required_list(required)
        eval_required = None if required is None else tuple(required)
        r_init = np.asarray(linear_problem.eval(required=eval_required), dtype=float).reshape(-1)
        initial_cost = float(r_init @ r_init)

    rnorm = float("inf")
    dxnorm = float("inf")

    def _outcome(
        *,
        status: str,
        iters: int,
        cost: float,
        rnorm_local: float,
        dxnorm_local: float,
        message: str = "",
    ) -> SolveOutcome:
        with prof.span("solve.finalize"):
            x_star = np.asarray(linear_problem.get_point(), dtype=float).reshape(-1).copy()
        return SolveOutcome(
            solution=x_star,
            stats=SolveStats(
                status=str(status),
                iterations=int(iters),
                initial_objective=float(initial_cost),
                objective=float(cost),
                residual_norm=float(rnorm_local),
                step_norm=float(dxnorm_local),
                message=str(message),
            ),
            timing=prof.snapshot(),
            meta={
                "solver": "gauss_newton",
                "x0": x0_start.copy(),
            },
        )

    for k in range(int(max_iters)):
        with prof.span("solve.iter.linearize"):
            r_all, J_all = linear_problem.linearize(required=req)
        rnorm = float(np.linalg.norm(r_all))

        if on_iter is not None:
            on_iter(k, rnorm, 0.0)

        if rnorm < tol_r:
            cost = float(r_all @ r_all)
            return _outcome(
                status="converged",
                iters=k,
                cost=cost,
                rnorm_local=rnorm,
                dxnorm_local=0.0,
            )

        cost_cur = float(r_all @ r_all)
        with prof.span("solve.iter.step"):
            lhs = J_all.T @ J_all
            damp = float(damping)
            if damp < 0.0:
                raise ValueError(f"solve_gauss_newton: damping must be >= 0, got {damp}.")
            if damp > 0.0:
                lhs = lhs + damp * np.eye(lhs.shape[0], dtype=float)
            rhs = -J_all.T @ r_all

            dx, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            dx = np.asarray(dx, dtype=float).reshape(-1)
            dxnorm = float(np.linalg.norm(dx))

        if not bool(line_search):
            if on_iter is not None:
                on_iter(k, rnorm, dxnorm)

            if dxnorm < tol_dx:
                cost = float(r_all @ r_all)
                return _outcome(
                    status="converged",
                    iters=k,
                    cost=cost,
                    rnorm_local=rnorm,
                    dxnorm_local=dxnorm,
                )

            with prof.span("solve.iter.update"):
                x_cur = np.asarray(linear_problem.get_point(), dtype=float).reshape(-1)
                linear_problem.set_point(x_cur + dx)
            continue

        beta = float(ls_beta)
        if not (0.0 < beta < 1.0):
            raise ValueError(f"solve_gauss_newton: ls_beta must be in (0,1), got {beta}.")
        min_step = float(ls_min_step)
        if min_step <= 0.0:
            raise ValueError(f"solve_gauss_newton: ls_min_step must be > 0, got {min_step}.")
        max_ls = int(ls_max_iters)
        if max_ls <= 0:
            raise ValueError(f"solve_gauss_newton: ls_max_iters must be > 0, got {max_ls}.")

        x_cur = np.asarray(linear_problem.get_point(), dtype=float).reshape(-1).copy()
        best_x = x_cur.copy()
        best_cost = cost_cur
        step = 1.0
        accepted = False

        with prof.span("solve.iter.linesearch"):
            for _ in range(max_ls):
                x_trial = x_cur + step * dx
                linear_problem.set_point(x_trial)
                r_trial = np.asarray(linear_problem.eval(required=eval_required), dtype=float).reshape(-1)
                cost_trial = float(r_trial @ r_trial)

                if cost_trial < best_cost:
                    best_cost = cost_trial
                    best_x = x_trial.copy()

                if cost_trial < cost_cur:
                    accepted = True
                    break

                step *= beta
                if step < min_step:
                    break

        linear_problem.set_point(best_x)
        dx_eff = np.asarray(best_x - x_cur, dtype=float).reshape(-1)
        dxnorm_eff = float(np.linalg.norm(dx_eff))
        dxnorm = dxnorm_eff

        if on_iter is not None:
            on_iter(k, rnorm, dxnorm_eff)

        if dxnorm_eff < tol_dx:
            return _outcome(
                status="converged",
                iters=k,
                cost=float(best_cost),
                rnorm_local=rnorm,
                dxnorm_local=dxnorm_eff,
            )

        if not accepted and dxnorm_eff == 0.0:
            return _outcome(
                status="stalled",
                iters=k,
                cost=float(best_cost),
                rnorm_local=rnorm,
                dxnorm_local=dxnorm_eff,
                message="line-search could not find an improving step.",
            )

    with prof.span("solve.final.linearize"):
        r_all = np.asarray(linear_problem.eval(required=eval_required), dtype=float).reshape(-1)
    rnorm = float(np.linalg.norm(r_all))
    cost = float(r_all @ r_all)
    return _outcome(
        status="max_iters",
        iters=int(max_iters),
        cost=cost,
        rnorm_local=rnorm,
        dxnorm_local=dxnorm,
    )


__all__ = [
    "solve_gauss_newton",
]
