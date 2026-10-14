from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ...core.outcome import SolveOutcome
from ...core.state_cache import StateKey
from ...core.timing import Profiler
from ._gauss_newton_loop import _solve_gauss_newton

Array = np.ndarray


def solve_gauss_newton_operator(
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
    inner_tol: float = 1e-8,
    inner_max_iters: int | None = None,
    line_search: bool = True,
    ls_beta: float = 0.5,
    ls_min_step: float = 1e-8,
    ls_max_iters: int = 12,
    ls_max_retries: int = 3,
    damping_increase: float = 10.0,
    damping_max: float = 1e12,
    damping_decrease: float = 0.1,
    damping_min_factor: float = 100.0,
    c_armijo: float = 1e-4,
    history: bool = True,
    history_vectors: bool = False,
    history_path: str | Path | None = None,
    line_search_history_path: str | Path | None = None,
    verbose: bool = True,
    on_iter: Callable[..., None] | None = None,
    profiler: Profiler | None = None,
    _step_solver: Callable[[Array, Any, float, Array], tuple[Array, dict[str, Any]]] | None = None,
    _damping_floor_fn: Callable[[Any], float] | None = None,
    _solver_name: str = "gauss_newton_operator",
    _solver_settings: dict[str, Any] | None = None,
) -> SolveOutcome:
    """Operator Gauss-Newton with the dense solver's globalization/history.

    Uses jvp/vjp and damped CGLS, never a stacked Jacobian in the solver.
    inner_tol is the relative normal-residual tolerance; inner_max_iters
    defaults to max(20, 2*n). An iteration limit retains the approximate step
    for the outer line search; inner diagnostics are recorded separately.
    The scale-aware damping floor is exact, computed by streaming the shorter
    of JVP columns and VJP rows. Providers can supply
    jacobian_column_squared_norms(required=...) to avoid this scan.
    Set damping_min_factor=0 to disable the floor and scan.
    Runtime expressions without JVP use local
    derivative blocks; this fallback is not fully matrix-free.

    The outer loop is shared with gauss_newton.py, including retries,
    roundoff acceptance, callbacks and gradient-based convergence.
    """

    return _solve_gauss_newton(
        problem=problem,
        max_iters=max_iters,
        x0=x0,
        required=required,
        weighted=weighted,
        term_indices=term_indices,
        tol_r=tol_r,
        tol_dx=tol_dx,
        tol_grad=tol_grad,
        damping=damping,
        inner_tol=inner_tol,
        inner_max_iters=inner_max_iters,
        line_search=line_search,
        ls_beta=ls_beta,
        ls_min_step=ls_min_step,
        ls_max_iters=ls_max_iters,
        ls_max_retries=ls_max_retries,
        damping_increase=damping_increase,
        damping_max=damping_max,
        damping_decrease=damping_decrease,
        damping_min_factor=damping_min_factor,
        c_armijo=c_armijo,
        history=history,
        history_vectors=history_vectors,
        history_path=history_path,
        line_search_history_path=line_search_history_path,
        verbose=verbose,
        on_iter=on_iter,
        profiler=profiler,
        _step_solver=_step_solver,
        _damping_floor_fn=_damping_floor_fn,
        _solver_name=_solver_name,
        _solver_settings=_solver_settings,
    )


__all__ = ["solve_gauss_newton_operator"]
