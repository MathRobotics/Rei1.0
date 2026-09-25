"""Damped least-squares directions globalized by Armijo backtracking."""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ...core.outcome import SolveOutcome, SolveStats
from ...core.state_cache import StateKey
from ...core.timing import Profiler, ensure_profiler
from ...problem import as_linearized_problem
from ...xops import as_vec
from ..history import SolverHistoryRecorder
from .levenberg_marquardt import _lm_step


def solve_lm_ls(
    problem: Any, max_iters: int = 200, *, x0: Any = None,
    required: Iterable[StateKey] | None = None, weighted: bool | None = None,
    term_indices: Iterable[int] | None = None, tol_grad: float = 1e-8,
    tau: float = 1e-3, damping: float | None = None,
    damping_min_factor: float = 100., damping_increase: float = 10.,
    damping_decrease: float = .1, damping_max: float = 1e12,
    ls_beta: float = .5, ls_min_step: float = 1e-12,
    ls_max_iters: int = 40, ls_max_retries: int = 3, c_armijo: float = 1e-4,
    history: bool = True, history_vectors: bool = False,
    history_path: str | Path | None = None,
    line_search_history_path: str | Path | None = None,
    verbose: bool = True, on_iter: Callable[..., None] | None = None,
    profiler: Profiler | None = None,
) -> SolveOutcome:
    """LM direction + Armijo search; only gradient tolerance certifies convergence.

    max_iters counts accepted updates; each update has bounded search retries.
    Full steps relax damping, shortened steps retain it, failed searches raise
    it and recompute the direction at the same point. No step-size convergence
    or roundoff/gradient-progress acceptance is used. Objective is ||r||².
    """
    for name, value, minimum in (("max_iters", max_iters, 0),
                                 ("ls_max_iters", ls_max_iters, 1),
                                 ("ls_max_retries", ls_max_retries, 0)):
        if not np.isfinite(value) or int(value) != value or value < minimum:
            raise ValueError(f"lm-ls: {name} must be an integer >= {minimum}.")
    if not np.isfinite(tol_grad) or tol_grad < 0:
        raise ValueError("lm-ls: tol_grad must be finite and >= 0.")
    for name, value in (("tau", tau), ("damping_min_factor", damping_min_factor),
                        ("damping_max", damping_max), ("ls_min_step", ls_min_step)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"lm-ls: {name} must be finite and > 0.")
    if damping is not None and (not np.isfinite(damping) or damping < 0):
        raise ValueError("lm-ls: damping must be finite and >= 0.")
    for name, value in (("ls_beta", ls_beta), ("c_armijo", c_armijo),
                        ("damping_decrease", damping_decrease)):
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"lm-ls: {name} must be in (0, 1).")
    if not np.isfinite(damping_increase) or damping_increase <= 1:
        raise ValueError("lm-ls: damping_increase must be finite and > 1.")
    if ls_min_step > 1:
        raise ValueError("lm-ls: ls_min_step must be <= 1.")

    prof = ensure_profiler(profiler)
    recorder = SolverHistoryRecorder(enabled=history, path=history_path,
        line_search_path=line_search_history_path, verbose=verbose, solver="lm-ls")
    model = as_linearized_problem(problem, weighted=weighted, term_indices=term_indices)
    if x0 is not None:
        model.set_point(as_vec(x0, expected_size=model.n_total, name="x0"))
    x = np.asarray(model.get_point(), dtype=float).reshape(-1).copy()
    start = x.copy()
    req = model.required_list(None if required is None else tuple(required))

    def linearize():
        with prof.span("solve.iter.linearize"):
            r, J = model.linearize(required=req)
        r, J = np.asarray(r, dtype=float).reshape(-1), np.asarray(J, dtype=float)
        if J.shape != (r.size, x.size):
            raise ValueError("lm-ls: inconsistent residual/Jacobian dimensions.")
        return r, J

    r, J = linearize()
    g, cost = J.T @ r, float(r @ r)
    if not all(np.all(np.isfinite(v)) for v in (x, r, J, g, cost)):
        raise ValueError("lm-ls: initial point, residual, Jacobian and objective must be finite.")
    initial_cost = cost

    def floor_and_scale():
        scale = float(np.max(np.sum(J * J, axis=0), initial=0.))
        floor = damping_min_factor * np.finfo(float).eps * scale
        if not np.isfinite(scale) or not np.isfinite(floor):
            raise ValueError("lm-ls: damping scale overflow; rescale the problem.")
        return floor, scale

    floor, scale = floor_and_scale()
    mu = max(floor, float(damping if damping is not None else tau * max(scale, np.finfo(float).tiny)))
    if not np.isfinite(mu):
        raise ValueError("lm-ls: initial damping overflow.")

    def fields():
        result = dict(objective=cost, residual_norm=float(np.linalg.norm(r)),
                      jt_r_inf_norm=float(np.max(np.abs(g), initial=0.)))
        if history_vectors:
            result.update(variables=x.tolist(), jt_r=g.tolist())
        return result

    def stationary():
        return bool(np.all(np.isfinite(g)) and np.max(np.abs(g), initial=0.) <= tol_grad)

    callback_with_gradient = False
    if on_iter is not None:
        try:
            inspect.signature(on_iter).bind(0, 0., 0., g.copy())
            callback_with_gradient = True
        except (TypeError, ValueError):
            pass
    settings = dict(max_iters=max_iters, tol_grad=tol_grad, tau=tau, damping=damping,
                    damping_min_factor=damping_min_factor, damping_increase=damping_increase,
                    damping_decrease=damping_decrease, damping_max=damping_max,
                    ls_beta=ls_beta, ls_min_step=ls_min_step, ls_max_iters=ls_max_iters,
                    ls_max_retries=ls_max_retries, c_armijo=c_armijo)
    recorder.emit("initial", 0, **fields(), step_norm=None, damping=mu,
                  damping_min=floor, settings=settings,
                  objective_definition="squared_residual_norm", variable_space="solver")
    iteration, step_norm = 0, 0.
    status, reason = "max_iters", "max_iters"
    while iteration < max_iters and not stationary():
        if on_iter is not None:
            args = (iteration, float(np.linalg.norm(r)), step_norm)
            on_iter(*args, g.copy()) if callback_with_gradient else on_iter(*args)
        floor, _ = floor_and_scale()
        mu = max(mu, floor)
        accepted, total_trials = False, 0
        reason = "line_search_retry_limit"
        for retry in range(int(ls_max_retries) + 1):
            with prof.span("solve.iter.step"):
                h = _lm_step(J, r, mu)
            alpha = 1.
            trial_reason = "nonfinite_step"
            for trial in range(1, int(ls_max_iters) + 1):
                if alpha < ls_min_step or not np.all(np.isfinite(h)):
                    break
                trial_x = x + alpha * h
                displacement = trial_x - x
                slope = float(2. * (g @ displacement))
                total_trials += 1
                trial_cost = None
                trial_reason = "unrepresentable_step" if np.array_equal(trial_x, x) else "not_descent"
                try:
                    if slope < 0 and np.all(np.isfinite(trial_x)):
                        model.set_point(trial_x)
                        with prof.span("solve.iter.trial"):
                            trial_r = np.asarray(model.eval(required=req), dtype=float).reshape(-1)
                        if trial_r.shape != r.shape:
                            raise ValueError("lm-ls: residual dimension changed at trial point.")
                        trial_cost = float(trial_r @ trial_r)
                        # Compare reductions, requiring strict improvement even
                        # if cost + c*slope rounds back to cost.
                        accepted = bool(np.isfinite(trial_cost) and cost - trial_cost > 0
                                        and cost - trial_cost >= -c_armijo * slope)
                        trial_reason = "accepted" if accepted else "insufficient_decrease"
                        if accepted:
                            new_r, new_J = linearize()
                            new_g = new_J.T @ new_r
                            new_cost = float(new_r @ new_r)
                            accepted = bool(all(np.all(np.isfinite(v)) for v in (new_r, new_J, new_g))
                                            and np.isfinite(new_cost) and cost - new_cost > 0
                                            and cost - new_cost >= -c_armijo * slope)
                            if not accepted:
                                trial_reason = "invalid_trial_linearization"
                except Exception as error:
                    recorder.emit("line_search_trial", iteration + 1, retry=retry,
                                  trial=trial, damping=mu, step_scale=alpha,
                                  objective=trial_cost, accepted=False,
                                  reason="evaluation_error", error=str(error))
                    raise
                finally:
                    model.set_point(x)
                recorder.emit("line_search_trial", iteration + 1, retry=retry, trial=trial,
                              damping=mu, damping_min=floor, step_scale=alpha,
                              step_norm=float(np.linalg.norm(displacement)), objective=trial_cost,
                              base_objective=cost, required_reduction=-c_armijo * slope,
                              accepted=accepted, reason=trial_reason,
                              **({"variables": trial_x.tolist()} if history_vectors else {}))
                if accepted or trial_reason == "unrepresentable_step":
                    break
                alpha *= ls_beta
            recorder.emit("line_search_end", iteration + 1, retry=retry, damping=mu,
                          accepted=accepted, reason=trial_reason)
            if accepted:
                break
            if trial_reason == "unrepresentable_step":
                reason = "unrepresentable_step_nonstationary"
                break
            cap = max(damping_max, floor)
            if mu >= cap:
                reason = "damping_limit_nonstationary"
                break
            if retry < ls_max_retries:
                previous_mu = mu
                mu = min(cap, mu * damping_increase)
                recorder.emit("line_search_retry", iteration + 1, retry=retry + 1,
                              previous_damping=previous_mu, damping=mu, reason="increase_damping")
        if not accepted:
            status = "stalled"
            break
        used_mu = mu
        x, r, J, g, cost = trial_x.copy(), new_r, new_J, new_g, new_cost
        model.set_point(x)
        iteration += 1
        step_norm = float(np.linalg.norm(displacement))
        floor, _ = floor_and_scale()
        mu = max(floor, mu * damping_decrease if alpha == 1. else mu)
        recorder.emit("iteration_end", iteration, **fields(), step_norm=step_norm,
                      damping=used_mu, next_damping=mu, step_scale=alpha,
                      line_search_trials=total_trials, line_search_status="accepted")
        reason = "max_iters"
    if stationary():
        status, reason = "converged", "gradient_tolerance"
    recorder.emit("final", iteration, **fields(), step_norm=step_norm, damping=mu,
                  status=status, reason=reason, gradient_converged=stationary())
    return SolveOutcome(solution=x.copy(), stats=SolveStats(
        status=status, iterations=iteration, initial_objective=initial_cost, objective=cost,
        residual_norm=float(np.linalg.norm(r)), step_norm=step_norm, message=reason),
        timing=prof.snapshot(), meta=dict(solver="lm-ls", reason=reason,
            gradient_converged=stationary(), x0=start), history=recorder.events,
        line_search_history=recorder.line_search_events)
