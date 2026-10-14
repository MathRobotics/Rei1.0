from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ...core.outcome import SolveOutcome
from ...core.state_cache import StateKey
from ...core.timing import Profiler
from ._gauss_newton_loop import _solve_gauss_newton

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
) -> SolveOutcome:
    """Gauss-Newton loop with accepted-state and line-search event history.

    Returns:
      SolveOutcome with converged/max-iter/stalled status and timing spans.

    History uses ||r||² (not half-cost), and Jᵀr in the solver's coordinates.
    ``history_path`` streams JSONL to a new file independently of in-memory
    ``history``. ``history_vectors`` includes variables and state Jᵀr vectors.
    Search details use ``line_search_history_path`` (default: a sibling
    <history stem>.line_search.jsonl) and ``outcome.line_search_history``.
    The legacy ``on_iter`` callback retains its pre-update semantics.
    State history is printed as it is recorded; use ``verbose=False`` to silence it.
    Failed nonstationary searches retry at most ``ls_max_retries`` times per
    outer iteration: expand the trial budget once, then increase damping.
    Successful full steps reduce damping, but never below
    ``damping_min_factor * eps * max(diag(J.T @ J))`` at the current point.
    Near roundoff, a trial must improve stationarity before it can be accepted.
    Convergence requires tol_grad.
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
        _dense=True, _solver_name="gauss_newton",
    )


__all__ = ["solve_gauss_newton"]
