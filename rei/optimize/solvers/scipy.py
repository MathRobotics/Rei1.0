"""SciPy solver integration with lazy optional dependency loading."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ...core.outcome import SolveOutcome, SolveStats
from ...core.state_cache import StateKey
from ...core.timing import Profiler, ensure_profiler
from ...problem import as_linearized_problem
from ...xops import as_vec
from ..history import BackendProgress
from ._external_common import (
    Array,
    IterCallback,
    SolveResult,
    _LinearizedObjective,
    _initialize_linear_problem_point,
    _extract_x_from_callback_arg,
)


def solve_scipy_minimize(
    problem: Any,
    *,
    x0: Array | Any = None,
    required: Iterable[StateKey] | None = None,
    weighted: bool | None = None,
    term_indices: Iterable[int] | None = None,
    method: str = "L-BFGS-B",
    max_iters: int | None = 200,
    tol: float | None = None,
    bounds: Any = None,
    verbose: bool = True,
    options: Mapping[str, Any] | None = None,
    on_iter: IterCallback | None = None,
    profiler: Profiler | None = None,
) -> SolveResult:
    """Solve `||r(x)||^2` via scipy.optimize.minimize and return SolveOutcome."""

    prof = ensure_profiler(profiler)
    try:
        from scipy.optimize import minimize
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "solve_scipy_minimize requires scipy. Install scipy and re-run."
        ) from e

    with prof.span("solve.setup"):
        linear_problem = as_linearized_problem(
            problem,
            weighted=weighted,
            term_indices=None if term_indices is None else tuple(int(i) for i in term_indices),
        )
        x0_init = _initialize_linear_problem_point(linear_problem, x0=x0)
        req = linear_problem.required_list(required)
        objective = _LinearizedObjective(linear_problem, required=req)
        n_total = int(x0_init.size)
        initial_cost, grad0, _rnorm0 = objective.eval(x0_init)
    progress = BackendProgress("scipy_minimize", verbose=verbose)
    progress.initial(initial_cost, grad0 / 2)

    options_local: dict[str, Any] = {} if options is None else dict(options)
    if max_iters is not None and "maxiter" not in options_local:
        options_local["maxiter"] = int(max_iters)

    iter_count = 0
    last_dxnorm = 0.0
    prev_x = x0_init.copy()

    def fun(x: Array) -> float:
        fx, _gx, _rnorm = objective.eval(x)
        return float(fx)

    def jac(x: Array) -> Array:
        _fx, gx, _rnorm = objective.eval(x)
        return gx

    def callback(xk: Any) -> None:
        nonlocal iter_count, last_dxnorm, prev_x
        x_vec = _extract_x_from_callback_arg(xk, size=n_total)
        if x_vec is None:
            return
        fx, gx, rnorm = objective.eval(x_vec)
        last_dxnorm = float(np.linalg.norm(x_vec - prev_x))
        prev_x = x_vec.copy()
        progress.iterate(fx, last_dxnorm, gx / 2)
        if on_iter is not None:
            on_iter(iter_count, rnorm, last_dxnorm)
        iter_count += 1

    with prof.span("solve.backend"):
        result = minimize(
            fun=fun,
            x0=x0_init,
            jac=jac,
            method=str(method),
            tol=tol,
            bounds=bounds,
            callback=callback,
            options=options_local,
        )

    with prof.span("solve.finalize"):
        x_star = as_vec(getattr(result, "x", x0_init), expected_size=n_total, name="result.x")
        cost, final_grad, rnorm = objective.eval(x_star)
    iters = int(getattr(result, "nit", iter_count))
    if iters <= 0:
        iters = int(iter_count)
    if iter_count <= 0:
        last_dxnorm = float(np.linalg.norm(x_star - x0_init))
    converged = bool(getattr(result, "success", False))
    status = "converged" if converged else "failed"
    message = str(getattr(result, "message", "") or "")

    outcome = SolveOutcome(
        solution=x_star.copy(),
        stats=SolveStats(
            status=status,
            iterations=iters,
            initial_objective=float(initial_cost),
            objective=float(cost),
            residual_norm=float(rnorm),
            step_norm=float(last_dxnorm),
            message=message,
        ),
        timing=prof.snapshot(),
        meta={
            "solver": "scipy_minimize",
            "method": str(method),
            "x0": x0_init.copy(),
        },
    )
    progress.final(outcome, final_grad / 2)
    return outcome


__all__ = ["solve_scipy_minimize"]
