from __future__ import annotations

import inspect
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
    weighted: bool | None = None,
    term_indices: Iterable[int] | None = None,
    tol_r: float = 1e-10,
    tol_dx: float = 1e-12,
    tol_grad: float = 1e-10,
    damping: float = 1e-8,
    line_search: bool = True,
    ls_beta: float = 0.5,
    ls_min_step: float = 1e-8,
    ls_max_iters: int = 12,
    on_iter: Callable[..., None] | None = None,
    profiler: Profiler | None = None,
) -> SolveOutcome:
    """Minimal Gauss-Newton loop for a linearized residual problem.

    Returns:
      SolveOutcome with converged/max-iter/stalled status and timing spans.
    """

    prof = ensure_profiler(profiler)
    # Validate before evaluating the model or changing x0. Invalid options must
    # not silently succeed just because the initial residual already vanishes.
    for name, value in (("tol_r", tol_r), ("tol_dx", tol_dx),
                        ("tol_grad", tol_grad), ("damping", damping)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"solve_gauss_newton: {name} must be finite and >= 0.")
    if not np.isfinite(max_iters) or int(max_iters) != max_iters or max_iters < 0:
        raise ValueError("solve_gauss_newton: max_iters must be a nonnegative integer.")
    if line_search:
        if not np.isfinite(ls_beta) or not 0 < ls_beta < 1:
            raise ValueError("solve_gauss_newton: ls_beta must be finite and in (0, 1).")
        if not np.isfinite(ls_min_step) or ls_min_step <= 0:
            raise ValueError("solve_gauss_newton: ls_min_step must be finite and > 0.")
        if not np.isfinite(ls_max_iters) or int(ls_max_iters) != ls_max_iters or ls_max_iters <= 0:
            raise ValueError("solve_gauss_newton: ls_max_iters must be a positive integer.")
    with prof.span("solve.setup"):
        linear_problem: LinearizedProblem = as_linearized_problem(
            problem,
            weighted=weighted,
            term_indices=None if term_indices is None else tuple(int(i) for i in term_indices),
        )
        n_total = int(linear_problem.n_total)
        if x0 is not None:
            linear_problem.set_point(as_vec(x0, expected_size=n_total, name="x0"))
        x0_start = np.asarray(linear_problem.get_point(), dtype=float).reshape(-1).copy()
        eval_required = None if required is None else tuple(required)
        req = linear_problem.required_list(eval_required)
        r_init = np.asarray(linear_problem.eval(required=eval_required), dtype=float).reshape(-1)
        initial_cost = float(r_init @ r_init)

    rnorm = float("inf")
    dxnorm = float("inf")

    def _on_iter_accepts_jt_r(callback: Callable[..., None]) -> bool:
        """Keep the established three-argument callback contract compatible."""
        try:
            inspect.signature(callback).bind(0, 0.0, 0.0, np.zeros((0,), dtype=float))
        except (TypeError, ValueError):
            return False
        return True

    callback_accepts_jt_r = on_iter is not None and _on_iter_accepts_jt_r(on_iter)

    def _emit_iteration(k: int, residual_norm: float, step_norm: float, jt_r: Array) -> None:
        if on_iter is None:
            return
        if callback_accepts_jt_r:
            on_iter(int(k), float(residual_norm), float(step_norm), np.asarray(jt_r, dtype=float).reshape(-1).copy())
            return
        on_iter(int(k), float(residual_norm), float(step_norm))

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

    def _finish_small_step(iters: int, step_norm: float) -> SolveOutcome:
        with prof.span("solve.final.linearize"):
            r, J = linear_problem.linearize(required=req)
        residual_norm = float(np.linalg.norm(r))
        stationary = float(np.max(np.abs(J.T @ r), initial=0.0)) <= tol_grad
        converged = residual_norm < tol_r or stationary
        return _outcome(
            status="converged" if converged else "stalled",
            iters=iters,
            cost=float(r @ r),
            rnorm_local=residual_norm,
            dxnorm_local=step_norm,
            message="" if converged else "step is small but residual and gradient tolerances are not met.",
        )

    for k in range(int(max_iters)):
        with prof.span("solve.iter.linearize"):
            r_all, J_all = linear_problem.linearize(required=req)
        rnorm = float(np.linalg.norm(r_all))
        jt_r = np.asarray(J_all.T @ r_all, dtype=float).reshape(-1)

        if rnorm < tol_r:
            _emit_iteration(k, rnorm, 0.0, jt_r)
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
            damp = float(damping)
            # Solve the residual system directly to avoid squaring its
            # condition number. Damping is an augmented least-squares term.
            lhs = np.asarray(J_all, dtype=float)
            rhs = -np.asarray(r_all, dtype=float)
            if damp > 0.0:
                lhs = np.vstack([lhs, np.sqrt(damp) * np.eye(lhs.shape[1])])
                rhs = np.concatenate([rhs, np.zeros(lhs.shape[1])])
            dx, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            dx = np.asarray(dx, dtype=float).reshape(-1)
            dxnorm = float(np.linalg.norm(dx))

        # A small computed step also needs a small gradient; otherwise apply
        # the step before checking convergence. Rejected trials are stalled.
        if dxnorm < tol_dx and float(np.max(np.abs(jt_r), initial=0.0)) <= tol_grad:
            _emit_iteration(k, rnorm, dxnorm, jt_r)
            return _outcome(
                status="converged",
                iters=k,
                cost=cost_cur,
                rnorm_local=rnorm,
                dxnorm_local=dxnorm,
            )

        if not bool(line_search):
            _emit_iteration(k, rnorm, dxnorm, jt_r)

            with prof.span("solve.iter.update"):
                x_cur = np.asarray(linear_problem.get_point(), dtype=float).reshape(-1)
                linear_problem.set_point(x_cur + dx)
            if dxnorm < tol_dx:
                return _finish_small_step(k + 1, dxnorm)
            continue

        beta = float(ls_beta)
        min_step = float(ls_min_step)
        max_ls = int(ls_max_iters)

        x_cur = np.asarray(linear_problem.get_point(), dtype=float).reshape(-1).copy()
        best_x = x_cur.copy()
        best_cost = cost_cur
        step = 1.0
        accepted = False

        try:
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
        finally:
            linear_problem.set_point(best_x)
        dx_eff = np.asarray(best_x - x_cur, dtype=float).reshape(-1)
        dxnorm_eff = float(np.linalg.norm(dx_eff))
        dxnorm = dxnorm_eff

        _emit_iteration(k, rnorm, dxnorm_eff, jt_r)

        if not accepted:
            return _outcome(
                status="stalled",
                iters=k,
                cost=float(best_cost),
                rnorm_local=rnorm,
                dxnorm_local=dxnorm_eff,
                message="line-search could not find an improving step.",
            )

        if dxnorm_eff < tol_dx:
            return _finish_small_step(k + 1, dxnorm_eff)

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
