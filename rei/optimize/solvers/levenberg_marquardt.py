"""Baseline LM: Madsen, Nielsen & Tingleff (2004), Algorithm 3.16.

Keep algorithmic extensions out of this baseline. The previous adaptive
Gauss-Newton/line-search implementation lives in gauss_newton.py unchanged.
"""
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


def _lm_step(J: np.ndarray, r: np.ndarray, damping: float) -> np.ndarray:
    """Solve (J.T J + lambda I) h = -J.T r without forming normal equations."""
    n = J.shape[1]
    lhs = np.vstack((J, np.sqrt(damping) * np.eye(n)))
    rhs = np.concatenate((-r, np.zeros(n)))
    return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _accepted_damping(damping: float, rho: float) -> float:
    # Clipping above 1 avoids cubing a huge ratio; the result is identical.
    factor = max(1. / 3., 1. - (2. * min(rho, 1.) - 1.) ** 3)
    return max(np.finfo(float).tiny, damping * factor)


def solve_levenberg_marquardt(
    problem: Any,
    max_iters: int = 200,
    *,
    x0: Any = None,
    required: Iterable[StateKey] | None = None,
    weighted: bool | None = None,
    term_indices: Iterable[int] | None = None,
    tol_grad: float = 1e-8,
    tol_dx: float = 1e-12,
    tau: float = 1e-3,
    damping: float | None = None,
    history: bool = True,
    history_vectors: bool = False,
    history_path: str | Path | None = None,
    trial_history_path: str | Path | None = None,
    verbose: bool = True,
    on_iter: Callable[..., None] | None = None,
    profiler: Profiler | None = None,
) -> SolveOutcome:
    """Classical gain-ratio LM with Nielsen's damping update (no line search).

    Initial lambda is tau * max(diag(J.T J)), unless damping is supplied.
    Accept iff rho > 0; otherwise retain x/J/r and increase lambda by nu,
    doubling nu on each rejection. Each trial counts toward max_iters.

    Stops on ||J.T r||inf <= tol_grad OR ||h|| <= tol_dx*(||x||+tol_dx).
    Step convergence does not certify stationarity; inspect gradient_converged
    in outcome.meta, or set tol_dx=0 to disable that stopping criterion.
    History/reduction quantities use ||r||², consistently (not half-cost).
    """
    if not np.isfinite(max_iters) or int(max_iters) != max_iters or max_iters < 0:
        raise ValueError("solve_levenberg_marquardt: max_iters must be a nonnegative integer.")
    for name, value in (("tol_grad", tol_grad), ("tol_dx", tol_dx)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"solve_levenberg_marquardt: {name} must be finite and >= 0.")
    for name, value in (("tau", tau), ("damping", damping)):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"solve_levenberg_marquardt: {name} must be finite and > 0.")

    prof = ensure_profiler(profiler)
    recorder = SolverHistoryRecorder(
        enabled=history, path=history_path, line_search_path=trial_history_path,
        verbose=verbose, solver="levenberg_marquardt", trial_kind="lm",
    )
    with prof.span("solve.setup"):
        model = as_linearized_problem(problem, weighted=weighted, term_indices=term_indices)
        if x0 is not None:
            model.set_point(as_vec(x0, expected_size=model.n_total, name="x0"))
        x = np.asarray(model.get_point(), dtype=float).reshape(-1).copy()
        start = x.copy()
        req = model.required_list(None if required is None else tuple(required))

    def linearize():
        with prof.span("solve.iter.linearize"):
            residual, jacobian = model.linearize(required=req)
        residual = np.asarray(residual, dtype=float).reshape(-1)
        jacobian = np.asarray(jacobian, dtype=float)
        if jacobian.shape != (residual.size, x.size):
            raise ValueError("LM: inconsistent residual/Jacobian dimensions.")
        return residual, jacobian

    r, J = linearize()
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(r)) and np.all(np.isfinite(J))):
        raise ValueError("LM: initial point, residual and Jacobian must be finite.")
    cost = initial_cost = float(r @ r)
    g = J.T @ r
    scale = float(np.max(np.sum(J * J, axis=0), initial=0.))
    mu = float(damping if damping is not None else tau * (scale if scale > 0 else 1.))
    if not (np.isfinite(cost) and np.all(np.isfinite(g)) and np.isfinite(mu) and mu > 0):
        raise ValueError("LM: initial objective, gradient or damping overflow/underflow; rescale the problem.")
    nu = 2.
    step_norm = 0.
    iteration = 0
    callback_with_gradient = False
    if on_iter is not None:
        try:
            inspect.signature(on_iter).bind(0, 0., 0., np.zeros(0))
            callback_with_gradient = True
        except (TypeError, ValueError):
            pass

    def fields():
        result = dict(objective=cost, residual_norm=float(np.linalg.norm(r)),
                      jt_r_inf_norm=float(np.max(np.abs(g), initial=0.)))
        if history_vectors:
            result.update(variables=x.tolist(), jt_r=g.tolist())
        return result

    def stationary():
        return bool(np.all(np.isfinite(g)) and np.max(np.abs(g), initial=0.) <= tol_grad)

    settings = dict(max_iters=int(max_iters), tol_grad=tol_grad, tol_dx=tol_dx,
                    tau=tau, damping=damping)
    recorder.emit("initial", 0, **fields(), step_norm=None, damping=mu,
                  objective_definition="squared_residual_norm", variable_space="solver",
                  algorithm="madsen_nielsen_tingleff_3.16", settings=settings)

    status, reason = "max_iters", "max_iters"
    for iteration in range(1, int(max_iters) + 1):
        if stationary():
            iteration -= 1
            status, reason = "converged", "gradient_tolerance"
            break
        if on_iter is not None:
            args = (iteration - 1, float(np.linalg.norm(r)), step_norm)
            on_iter(*args, g.copy()) if callback_with_gradient else on_iter(*args)
        with prof.span("solve.iter.step"):
            h = _lm_step(J, r, mu)
        proposed_norm = float(np.linalg.norm(h))
        if not np.all(np.isfinite(h)):
            status, reason = "failed", "nonfinite_step"
            break
        if tol_dx > 0 and proposed_norm <= tol_dx * (float(np.linalg.norm(x)) + tol_dx):
            status, reason = "converged", "step_tolerance"
            break
        trial_x = x + h
        if np.array_equal(trial_x, x):
            status, reason = "stalled", "unrepresentable_step"
            break

        # Predict the reduction for the displacement actually representable.
        h = trial_x - x
        Jh = J @ h
        predicted = float(-2. * (g @ h) - Jh @ Jh)
        base_cost = cost
        trial_cost, actual, rho = float("inf"), float("-inf"), float("-inf")
        accepted = False
        trial_reason = "nonpositive_prediction"
        try:
            if np.isfinite(predicted) and predicted > 0 and np.all(np.isfinite(trial_x)):
                model.set_point(trial_x)
                with prof.span("solve.iter.trial"):
                    trial_r = np.asarray(model.eval(required=req), dtype=float).reshape(-1)
                if trial_r.shape != r.shape:
                    raise ValueError("LM: residual dimension changed at trial point.")
                trial_cost = float(trial_r @ trial_r)
                actual = base_cost - trial_cost
                rho = actual / predicted
                accepted = bool(np.isfinite(trial_cost) and np.isfinite(rho) and rho > 0)
                trial_reason = "accepted" if accepted else "no_decrease"
                if not np.isfinite(trial_cost):
                    trial_reason = "nonfinite_objective"
                if accepted:
                    new_r, new_J = linearize()
                    new_g = new_J.T @ new_r
                    if not (np.all(np.isfinite(new_r)) and np.all(np.isfinite(new_J))
                            and np.all(np.isfinite(new_g))):
                        accepted = False
                        trial_reason = "nonfinite_linearization"
                    else:
                        x, r, J, g = trial_x.copy(), new_r, new_J, new_g
                        cost = float(r @ r)
        finally:
            # Also restore the accepted point when evaluation raises.
            model.set_point(x)

        previous_mu = mu
        if accepted:
            mu, nu = _accepted_damping(mu, rho), 2.
            step_norm = float(np.linalg.norm(h))
        else:
            mu, nu = mu * nu, 2. * nu
            step_norm = 0.
        recorder.emit("lm_trial", iteration, trial=iteration, objective=trial_cost,
                      base_objective=base_cost, predicted_reduction=predicted,
                      actual_reduction=actual, gain_ratio=rho, damping=previous_mu,
                      next_damping=mu, step_norm=float(np.linalg.norm(h)),
                      accepted=accepted, reason=trial_reason,
                      **({"variables": trial_x.tolist()} if history_vectors else {}))
        recorder.emit("iteration_end", iteration, **fields(), step_norm=step_norm,
                      damping=previous_mu, next_damping=mu, gain_ratio=rho,
                      accepted=accepted, reason=trial_reason)
        if not np.isfinite(mu):
            status, reason = "stalled", "damping_overflow"
            break

    if stationary():
        status, reason = "converged", "gradient_tolerance"
    recorder.emit("final", iteration, **fields(), step_norm=step_norm, damping=mu,
                  status=status, reason=reason, gradient_converged=stationary())
    return SolveOutcome(
        solution=x,
        stats=SolveStats(status=status, iterations=iteration, initial_objective=initial_cost,
                         objective=cost, residual_norm=float(np.linalg.norm(r)),
                         step_norm=step_norm, message=reason),
        timing=prof.snapshot(),
        meta={"solver": "levenberg_marquardt", "algorithm": "madsen_nielsen_tingleff_3.16",
              "x0": start, "settings": settings, "reason": reason,
              "gradient_converged": stationary()},
        history=recorder.events, trial_history=recorder.line_search_events,
    )
