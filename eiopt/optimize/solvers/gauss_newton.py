from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from ...core.state_cache import StateKey
from .._xops import apply_pack_dx, set_pack_x
from ..runtime import NLSRuntime

Array = np.ndarray

def solve_gauss_newton(
    runtime: NLSRuntime,
    max_iters: int = 20,
    *,
    required: Iterable[StateKey] | None = None,
    tol_r: float = 1e-10,
    tol_dx: float = 1e-12,
    damping: float = 1e-8,
    line_search: bool = True,
    ls_beta: float = 0.5,
    ls_min_step: float = 1e-8,
    ls_max_iters: int = 12,
    on_iter: Callable[[int, float, float], None] | None = None,
) -> tuple[Array, float, float, int, float, float, bool]:
    """Minimal Gauss-Newton loop for a `NLSRuntime`.

    Returns:
      (x_star, initial_cost, cost, iters, rnorm, dxnorm, converged)
    """

    rnorm = float("inf")
    dxnorm = float("inf")
    converged = False

    req = runtime.required_list(required)
    r_init, _J_init = runtime.linearize(required=req)
    initial_cost = float(r_init @ r_init)

    variables = runtime.pack

    for k in range(int(max_iters)):
        r_all, J_all = runtime.linearize(required=req)
        rnorm = float(np.linalg.norm(r_all))

        if on_iter is not None:
            on_iter(k, rnorm, 0.0)

        if rnorm < tol_r:
            converged = True
            x_star = np.asarray(variables.get(), dtype=float).reshape(-1).copy()
            cost = float(r_all @ r_all)
            return x_star, initial_cost, cost, k, rnorm, 0.0, converged

        cost_cur = float(r_all @ r_all)
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
                converged = True
                x_star = np.asarray(variables.get(), dtype=float).reshape(-1).copy()
                cost = float(r_all @ r_all)
                return x_star, initial_cost, cost, k, rnorm, dxnorm, converged

            apply_pack_dx(variables, dx, name="dx")
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

        x_cur = np.asarray(variables.get(), dtype=float).reshape(-1).copy()
        best_x = x_cur.copy()
        best_cost = cost_cur
        step = 1.0
        accepted = False

        for _ in range(max_ls):
            x_trial = x_cur + step * dx
            set_pack_x(variables, x_trial, name="x")
            r_trial, _ = runtime.linearize(required=req)
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

        set_pack_x(variables, best_x, name="x")
        dx_eff = np.asarray(best_x - x_cur, dtype=float).reshape(-1)
        dxnorm_eff = float(np.linalg.norm(dx_eff))
        dxnorm = dxnorm_eff

        if on_iter is not None:
            on_iter(k, rnorm, dxnorm_eff)

        if dxnorm_eff < tol_dx:
            converged = True
            x_star = np.asarray(variables.get(), dtype=float).reshape(-1).copy()
            cost = float(best_cost)
            return x_star, initial_cost, cost, k, rnorm, dxnorm_eff, converged

        if not accepted and dxnorm_eff == 0.0:
            x_star = np.asarray(variables.get(), dtype=float).reshape(-1).copy()
            cost = float(best_cost)
            return x_star, initial_cost, cost, k, rnorm, dxnorm_eff, converged

    r_all, _J_all = runtime.linearize(required=req)
    rnorm = float(np.linalg.norm(r_all))
    x_star = np.asarray(variables.get(), dtype=float).reshape(-1).copy()
    cost = float(r_all @ r_all)
    return x_star, initial_cost, cost, int(max_iters), rnorm, dxnorm, converged


__all__ = [
    "solve_gauss_newton",
]
