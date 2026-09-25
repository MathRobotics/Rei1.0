from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ...core.outcome import SolveOutcome, SolveStats
from ...core.state_cache import StateKey
from ...core.timing import Profiler, ensure_profiler
from ...problem import LinearizedProblem, as_linearized_problem
from ...xops import as_vec
from ..history import SolverHistoryRecorder

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
    Successful full steps reduce damping. Near roundoff, a trial must improve
    stationarity before it can be accepted. Convergence requires tol_grad.
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
        if not np.isfinite(damping_decrease) or not 0 < damping_decrease < 1:
            raise ValueError("solve_gauss_newton: damping_decrease must be finite and in (0, 1).")
        if not np.isfinite(c_armijo) or not 0 < c_armijo < 1:
            raise ValueError("solve_gauss_newton: c_armijo must be finite and in (0, 1).")
        if not np.isfinite(ls_max_retries) or int(ls_max_retries) != ls_max_retries or ls_max_retries < 0:
            raise ValueError("solve_gauss_newton: ls_max_retries must be a nonnegative integer.")
        if not np.isfinite(damping_increase) or damping_increase <= 1:
            raise ValueError("solve_gauss_newton: damping_increase must be finite and > 1.")
        if not np.isfinite(damping_max) or damping_max <= 0:
            raise ValueError("solve_gauss_newton: damping_max must be finite and > 0.")
        if not np.isfinite(ls_beta) or not 0 < ls_beta < 1:
            raise ValueError("solve_gauss_newton: ls_beta must be finite and in (0, 1).")
        if not np.isfinite(ls_min_step) or ls_min_step <= 0:
            raise ValueError("solve_gauss_newton: ls_min_step must be finite and > 0.")
        if not np.isfinite(ls_max_iters) or int(ls_max_iters) != ls_max_iters or ls_max_iters <= 0:
            raise ValueError("solve_gauss_newton: ls_max_iters must be a positive integer.")
    recorder = SolverHistoryRecorder(enabled=history, path=history_path,
                                     line_search_path=line_search_history_path, verbose=verbose)
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
    dxnorm = 0.0
    last_recorded_iteration = -1
    current_damping = float(damping)
    pending_linearization: tuple[Array, Array] | None = None
    search_summary: dict[str, Any] = {
        "line_search_trials": 0, "step_scale": None, "line_search_status": "disabled",
    }

    def _point_fields(r: Array, J: Array) -> dict[str, Any]:
        jt_r = np.asarray(J.T @ r, dtype=float).reshape(-1)
        fields: dict[str, Any] = {
            "objective": float(r @ r),
            "residual_norm": float(np.linalg.norm(r)),
            "jt_r_inf_norm": float(np.max(np.abs(jt_r), initial=0.0)),
        }
        if history_vectors:
            fields["jt_r"] = jt_r.tolist()
            fields["variables"] = np.asarray(linear_problem.get_point()).reshape(-1).tolist()
        return fields

    def _record_point(k: int, r: Array, J: Array, step_norm: float) -> None:
        nonlocal last_recorded_iteration
        if recorder.active and k > last_recorded_iteration:
            metadata = {} if k else {
                "objective_definition": "squared_residual_norm",
                "variable_space": "solver",
                "settings": {
                    "max_iters": int(max_iters), "tol_r": tol_r,
                    "tol_dx": tol_dx, "tol_grad": tol_grad, "damping": damping,
                    "line_search": bool(line_search), "ls_beta": ls_beta,
                    "ls_min_step": ls_min_step, "ls_max_iters": int(ls_max_iters),
                    "ls_max_retries": int(ls_max_retries),
                    "damping_increase": damping_increase, "damping_max": damping_max,
                    "damping_decrease": damping_decrease, "c_armijo": c_armijo,
                },
            }
            recorder.emit(
                "initial" if k == 0 else "iteration_end", k,
                **_point_fields(r, J), step_norm=None if k == 0 else step_norm,
                **metadata,
                **(search_summary if k else {}),
            )
            last_recorded_iteration = k

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
        reason: str | None = None,
        final_linearization: tuple[Array, Array] | None = None,
    ) -> SolveOutcome:
        with prof.span("solve.finalize"):
            x_star = np.asarray(linear_problem.get_point(), dtype=float).reshape(-1).copy()
            if recorder.active:
                r_final, J_final = (
                    linear_problem.linearize(required=req)
                    if final_linearization is None else final_linearization
                )
                r_final = np.asarray(r_final, dtype=float).reshape(-1)
                fields = _point_fields(r_final, J_final)
                cost = fields["objective"]
                rnorm_local = fields["residual_norm"]
                _record_point(iters, r_final, J_final, dxnorm_local)
                recorder.emit("final", iters, **fields, step_norm=dxnorm_local,
                              status=status, reason=reason or status, message=message)
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
            history=recorder.events,
            line_search_history=recorder.line_search_events,
        )

    def _stationary(gradient: Array) -> bool:
        return bool(np.all(np.isfinite(gradient))
                    and float(np.max(np.abs(gradient), initial=0.0)) <= tol_grad)

    def _direction(r: Array, J: Array, damp: float) -> Array:
        with prof.span("solve.iter.step"):
            lhs = np.asarray(J, dtype=float)
            rhs = -np.asarray(r, dtype=float)
            if damp > 0.0:
                lhs = np.vstack([lhs, np.sqrt(damp) * np.eye(lhs.shape[1])])
                rhs = np.concatenate([rhs, np.zeros(lhs.shape[1])])
            dx, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            return np.asarray(dx, dtype=float).reshape(-1)

    for k in range(int(max_iters)):
        with prof.span("solve.iter.linearize"):
            r_all, J_all = (linear_problem.linearize(required=req)
                            if pending_linearization is None else pending_linearization)
            pending_linearization = None
        rnorm = float(np.linalg.norm(r_all))
        jt_r = np.asarray(J_all.T @ r_all, dtype=float).reshape(-1)
        _record_point(k, r_all, J_all, dxnorm)

        if _stationary(jt_r):
            _emit_iteration(k, rnorm, 0.0, jt_r)
            cost = float(r_all @ r_all)
            return _outcome(
                status="converged",
                iters=k,
                cost=cost,
                rnorm_local=rnorm,
                dxnorm_local=0.0,
                reason="residual_tolerance" if rnorm < tol_r else "gradient_tolerance",
                final_linearization=(r_all, J_all),
            )

        cost_cur = float(r_all @ r_all)
        # Augmented least squares avoids explicitly forming J.T @ J.
        dx = _direction(r_all, J_all, current_damping)
        dxnorm = float(np.linalg.norm(dx))

        if not bool(line_search):
            _emit_iteration(k, rnorm, dxnorm, jt_r)

            with prof.span("solve.iter.update"):
                x_cur = np.asarray(linear_problem.get_point(), dtype=float).reshape(-1)
                x_next = x_cur + dx
                if np.array_equal(x_next, x_cur):
                    return _outcome(
                        status="stalled", iters=k, cost=cost_cur,
                        rnorm_local=rnorm, dxnorm_local=0.0,
                        reason="unrepresentable_step",
                        message="update cannot change the point at floating-point precision.",
                        final_linearization=(r_all, J_all),
                    )
                linear_problem.set_point(x_next)
                dxnorm = float(np.linalg.norm(x_next - x_cur))
            continue

        beta = float(ls_beta)
        min_step = float(ls_min_step)
        max_ls = int(ls_max_iters)
        x_cur = np.asarray(linear_problem.get_point(), dtype=float).reshape(-1).copy()
        accepted = False
        stationary = False
        total_trials = 0
        retry = 0
        expanded = False
        reduced_damping = False
        stop_reason = None

        while True:
            best_x = x_cur.copy()
            best_cost = cost_cur
            step = 1.0
            search_reason = "max_trials"
            trial_count = 0
            settings = {
                "retry": retry, "damping": current_damping,
                "ls_max_iters": max_ls, "ls_beta": beta, "ls_min_step": min_step,
                "c_armijo": c_armijo,
            }
            try:
                with prof.span("solve.iter.linesearch"):
                    for trial in range(1, max_ls + 1):
                        if step < min_step:
                            search_reason = "step_too_small"
                            break
                        trial_count = trial
                        total_trials += 1
                        x_trial = x_cur + step * dx
                        linear_problem.set_point(x_trial)
                        try:
                            r_trial = np.asarray(linear_problem.eval(required=eval_required), dtype=float).reshape(-1)
                        except Exception as error:
                            recorder.emit("line_search_trial", k + 1, **settings, trial=trial,
                                          step_scale=step, objective=None, accepted=False,
                                          reason="evaluation_error", error=str(error))
                            recorder.emit("line_search_end", k + 1, **settings, trials=trial,
                                          accepted=False, reason="evaluation_error")
                            linear_problem.set_point(x_cur)
                            recorder.emit("iteration_failed", k + 1,
                                          **_point_fields(r_all, J_all), step_norm=0.0,
                                          line_search_trials=total_trials, step_scale=None,
                                          line_search_status="evaluation_error", **settings)
                            raise
                        cost_trial = float(r_trial @ r_trial)
                        displacement = x_trial - x_cur
                        trial_step_norm = float(np.linalg.norm(displacement))
                        noise_floor = 8 * np.finfo(float).eps * max(abs(cost_cur), abs(cost_trial))
                        reduction = cost_cur - cost_trial
                        required_reduction = -2 * c_armijo * float(jt_r @ displacement)
                        meaningful_decrease = bool(
                            np.isfinite(cost_trial) and required_reduction > 0
                            and reduction > noise_floor and reduction >= required_reduction
                        )
                        trial_accepted = meaningful_decrease
                        trial_reason = "accepted" if trial_accepted else "insufficient_decrease"
                        trial_gradient_norm = None
                        trial_linearization = None
                        if not np.isfinite(cost_trial):
                            trial_reason = "non_finite"
                        elif abs(reduction) <= noise_floor and np.any(displacement != 0):
                            # The objective cannot resolve this change. Accept only
                            # independently verified progress toward stationarity.
                            r_check, J_check = linear_problem.linearize(required=req)
                            r_check = np.asarray(r_check, dtype=float).reshape(-1)
                            g_check = np.asarray(J_check.T @ r_check).reshape(-1)
                            trial_gradient_norm = float(np.max(np.abs(g_check), initial=0.0))
                            checked_cost = float(r_check @ r_check)
                            trial_accepted = bool(
                                np.isfinite(checked_cost) and checked_cost <= cost_cur + noise_floor
                                and np.all(np.isfinite(g_check))
                                and (_stationary(g_check) or trial_gradient_norm <=
                                     0.5 * float(np.max(np.abs(jt_r), initial=0.0)))
                            )
                            trial_reason = "gradient_progress" if trial_accepted else "roundoff_no_gradient_progress"
                            if trial_accepted:
                                r_trial, cost_trial = r_check, checked_cost
                                trial_linearization = (r_check, J_check)
                        elif not np.any(displacement != 0):
                            trial_reason = "unrepresentable_step"
                        trial_fields: dict[str, Any] = {
                            **settings, "trial": trial, "step_scale": step,
                            "step_norm": trial_step_norm,
                            "objective": cost_trial, "base_objective": cost_cur,
                            "residual_norm": float(np.linalg.norm(r_trial)),
                            "accepted": trial_accepted,
                            "reason": trial_reason,
                            "required_reduction": required_reduction,
                            "objective_resolution": noise_floor,
                            "jt_r_inf_norm": trial_gradient_norm,
                        }
                        if history_vectors:
                            trial_fields["variables"] = x_trial.tolist()
                        recorder.emit("line_search_trial", k + 1, **trial_fields)
                        if trial_accepted:
                            best_cost = cost_trial
                            best_x = x_trial.copy()
                            accepted = True
                            search_reason = "accepted"
                            pending_linearization = trial_linearization
                            break
                        if (np.isfinite(cost_trial) and trial_step_norm <= tol_dx
                                and abs(reduction) <= noise_floor):
                            search_reason = "tiny_update"
                            break
                        step *= beta
                        if step < min_step:
                            search_reason = "step_too_small"
                            break
            finally:
                linear_problem.set_point(best_x)

            dxnorm_eff = float(np.linalg.norm(best_x - x_cur))
            recorder.emit("line_search_end", k + 1, **settings, trials=trial_count,
                          accepted=accepted, reason=search_reason,
                          step_scale=step if accepted else None,
                          step_norm=dxnorm_eff, objective=float(best_cost))
            if accepted:
                break

            # Judge stationarity at the restored point, not at a rejected trial.
            with prof.span("solve.final.linearize"):
                r_all, J_all = linear_problem.linearize(required=req)
            r_all = np.asarray(r_all, dtype=float).reshape(-1)
            jt_r = np.asarray(J_all.T @ r_all, dtype=float).reshape(-1)
            cost_cur = float(r_all @ r_all)
            rnorm = float(np.linalg.norm(r_all))
            stationary = _stationary(jt_r)
            if stationary:
                break
            if not np.all(np.isfinite(jt_r)):
                stop_reason = "non_finite_gradient"
                break
            if retry >= int(ls_max_retries):
                stop_reason = "retry_limit" if ls_max_retries else search_reason
                break

            previous_damping = current_damping
            previous_max_ls = max_ls
            if search_reason == "tiny_update" and current_damping > 0 and not reduced_damping:
                current_damping *= damping_decrease
                reduced_damping = True
                retry_reason = "decrease_damping"
            elif not expanded and search_reason == "max_trials":
                # First widen the search without changing its direction.
                max_ls *= 2
                expanded = True
                retry_reason = "expand_search"
            else:
                # If shrinking is insufficient, regularize and recompute the direction.
                if current_damping >= damping_max:
                    stop_reason = "damping_limit"
                    break
                next_damping = min(
                    damping_max, max(1e-6, current_damping * damping_increase),
                )
                current_damping = float(next_damping)
                retry_reason = "increase_damping"
            retry += 1
            retry_fields = {
                "retry": retry, "reason": retry_reason,
                "previous_damping": previous_damping, "damping": current_damping,
                "previous_ls_max_iters": previous_max_ls, "ls_max_iters": max_ls,
                "ls_beta": beta, "ls_min_step": min_step,
                "line_search_trials": total_trials,
            }
            recorder.emit("line_search_retry", k + 1, **retry_fields)
            recorder.emit("iteration_retry", k + 1,
                          **_point_fields(r_all, J_all), step_norm=0.0, **retry_fields)
            if retry_reason in {"increase_damping", "decrease_damping"}:
                dx = _direction(r_all, J_all, current_damping)

        dxnorm = dxnorm_eff
        search_summary = {
            "line_search_trials": total_trials, "retry": retry,
            "step_scale": step if accepted else None,
            "line_search_status": search_reason,
            "acceptance_reason": trial_reason if accepted else None,
            "damping": current_damping, "ls_max_iters": max_ls,
        }
        _emit_iteration(k, rnorm, dxnorm_eff, jt_r)

        if not accepted:
            recorder.emit("iteration_failed", k + 1,
                          **_point_fields(r_all, J_all), step_norm=0.0, **search_summary)
            return _outcome(
                status="converged" if stationary else "stalled",
                iters=k,
                cost=float(r_all @ r_all),
                rnorm_local=float(np.linalg.norm(r_all)),
                dxnorm_local=dxnorm_eff,
                message=(
                    "line-search found no improving step, but the restored point satisfies gradient tolerance."
                    if stationary else "line-search could not find an improving step."
                ),
                reason=("gradient_tolerance_after_line_search_failure"
                        if stationary else "line_search_" + str(stop_reason or search_reason)),
                final_linearization=(r_all, J_all),
            )

        # Keep making progress even if a damped accepted step is small. A good
        # full step restores larger directions by relaxing the regularization.
        if step == 1.0 or dxnorm_eff <= tol_dx:
            current_damping *= damping_decrease
        search_summary["next_damping"] = current_damping

    with prof.span("solve.final.linearize"):
        r_all, J_all = (linear_problem.linearize(required=req)
                        if pending_linearization is None else pending_linearization)
        r_all = np.asarray(r_all, dtype=float).reshape(-1)
    stationary = _stationary(np.asarray(J_all.T @ r_all).reshape(-1))
    rnorm = float(np.linalg.norm(r_all))
    cost = float(r_all @ r_all)
    return _outcome(
        status="converged" if stationary else "max_iters",
        iters=int(max_iters),
        cost=cost,
        rnorm_local=rnorm,
        dxnorm_local=dxnorm,
        reason="gradient_tolerance" if stationary else "max_iters",
        final_linearization=(r_all, J_all),
    )


__all__ = [
    "solve_gauss_newton",
]
