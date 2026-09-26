"""Inexact Gauss-Newton with a trust region and preconditioned truncated CG.

All nonlinear curvature products use JVP/VJP. No column-norm scan or normal
matrix is needed. A small affine-residual Gram matrix may precondition DOC.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ...core.outcome import SolveOutcome, SolveStats
from ...core.state_cache import StateKey
from ...core.timing import Profiler, ensure_profiler
from ...xops import as_vec
from ..history import SolverHistoryRecorder
from ._jacobian_operator import OperatorLinearizationProblem, _vector
from ._jacobian_operator import cgls_step
from .gauss_newton_operator import solve_gauss_newton_operator


def _boundary(h, p, radius):
    """Forward intersection h + tau*p with the Euclidean trust sphere."""
    pp = float(p @ p)
    hp = float(h @ p)
    remaining = max(0., (radius - np.linalg.norm(h)) * (radius + np.linalg.norm(h)))
    root = np.sqrt(hp*hp + pp*remaining)
    tau = remaining / (root + hp) if hp > 0 else (root - hp) / pp
    return h + tau*p


def _truncated_cg(J, g, radius, apply_inverse, *, tolerance, max_iters):
    """Steihaug PCG, with independently checked interior normal residuals."""
    n = g.size
    h = np.zeros(n)
    residual = g.copy()
    z = _vector(apply_inverse(residual), n, 'preconditioner output')
    rz = float(residual @ z)
    p = -z
    target = tolerance * float(np.linalg.norm(g))
    status = 'max_iters'
    iterations = 0
    if not np.isfinite(rz) or rz <= 0:
        raise ValueError('Krylov GN requires a positive-definite preconditioner.')
    for iterations in range(1, max_iters + 1):
        Jp = J @ p
        curvature = float(Jp @ Jp)
        if not np.isfinite(curvature):
            raise ValueError('Krylov GN: non-finite curvature; rescale the problem.')
        if curvature <= 0:
            h = _boundary(h, p, radius)
            status = 'zero_curvature'
            break
        alpha = rz / curvature
        trial = h + alpha*p
        if not np.all(np.isfinite(trial)) or np.linalg.norm(trial) >= radius:
            h = _boundary(h, p, radius)
            status = 'boundary'
            break
        h = trial
        residual += alpha * (J.T @ Jp)
        checked = False
        if np.linalg.norm(residual) <= target or iterations == max_iters:
            # Recompute before reporting convergence, rather than relying on a
            # cancellation-prone recursively updated CG residual.
            Jh = J @ h
            residual = g + J.T @ Jh
            checked = True
            if np.linalg.norm(residual) <= target:
                status = 'converged'
                return h, Jh, dict(iterations=iterations, status=status,
                    relative_normal_residual=float(np.linalg.norm(residual)/np.linalg.norm(g)))
        z = _vector(apply_inverse(residual), n, 'preconditioner output')
        next_rz = float(residual @ z)
        if not np.isfinite(next_rz) or next_rz <= 0:
            raise ValueError('Krylov GN: invalid preconditioned residual.')
        p = -z if checked else -z + (next_rz/rz)*p
        rz = next_rz
    Jh = J @ h
    # Boundary and iteration-limit exits are deliberately inexact, not inner
    # convergence. Avoid spending a VJP merely to diagnose a boundary step.
    return h, Jh, dict(iterations=iterations, status=status,
                       relative_normal_residual=None)


def _preconditioner(model, J, mode, *, probes, max_size, floor, rng):
    n = J.shape[1]
    if mode in ('auto', 'linear'):
        provider = getattr(model, 'linear_residual_gram', None)
        gram = provider(max_size=max_size) if callable(provider) and n <= max_size else None
        if gram is not None:
            gram = np.asarray(gram, dtype=float)
            if gram.shape != (n, n) or not np.all(np.isfinite(gram)):
                raise ValueError('Linear residual Gram must be finite and n_total by n_total.')
            if np.max(np.abs(gram-gram.T), initial=0.) > 1e-10*np.max(np.abs(gram), initial=0.):
                raise ValueError('Linear residual Gram must be symmetric.')
            eigenvalues, vectors = np.linalg.eigh((gram+gram.T)*.5)
            scale = float(np.max(eigenvalues, initial=0.))
            if float(np.min(eigenvalues, initial=0.)) < -1e-10 * scale:
                raise ValueError('Linear residual Gram must be positive semidefinite.')
            if scale > 0:
                inverse = 1. / np.maximum(eigenvalues, floor*scale)
                return lambda v: vectors @ (inverse * (vectors.T @ v)), 'linear'
        if mode == 'linear':
            raise ValueError('No nonzero affine-residual preconditioner within the size budget.')
    if mode == 'identity' or not n:
        return lambda v: v.copy(), 'identity'
    # E[(J.T w)**2] = diag(J.T J) for independent Rademacher residual probes.
    # This is ONLY a preconditioner estimate, never a damping/convergence bound.
    diagonal = np.zeros(n)
    for _ in range(probes):
        w = rng.choice(np.array([-1., 1.]), size=J.shape[0])
        product = J.T @ w
        diagonal += product*product/probes
    scale = float(np.max(diagonal, initial=0.))
    if not np.isfinite(scale):
        raise ValueError('Diagonal preconditioner overflow; rescale the problem.')
    if scale == 0:
        return lambda v: v.copy(), 'identity'
    diagonal = np.maximum(diagonal, floor*scale)
    return lambda v: v/diagonal, 'diagonal'


def _solve_gauss_newton_krylov_trust_region(
    problem: Any, max_iters: int = 200, *, x0: Any = None,
    required: Iterable[StateKey] | None = None, weighted: bool | None = None,
    term_indices: Iterable[int] | None = None, tol_grad: float = 1e-8,
    initial_radius: float | None = None, max_radius: float = 1e8,
    acceptance: float = .1, inner_max_iters: int = 50,
    forcing_min: float = 1e-4, forcing_max: float = .1,
    preconditioner: str = 'auto', preconditioner_probes: int = 8,
    preconditioner_max_size: int = 512, preconditioner_floor: float = 1e-10,
    preconditioner_refresh: int = 5, seed: int = 0,
    history: bool = True, history_vectors: bool = False,
    history_path: str | Path | None = None, trial_history_path: str | Path | None = None,
    verbose: bool = True, on_iter: Callable[..., None] | None = None,
    profiler: Profiler | None = None,
) -> SolveOutcome:
    """JVP-native trust-region GN; max_iters counts trials including rejections.

    Convergence is exclusively ||J.T r||inf <= tol_grad. PCG tolerance tightens
    toward the solution. Default preconditioning uses cheap affine residuals
    if available (size limited), otherwise a fixed number of VJP probes.
    No full-column damping scale or nonlinear Jacobian is assembled. The
    accepted-point objective and gradient are always checked, including when
    objective improvement is below floating-point resolution.
    """
    for name, value, minimum in (
        ('max_iters', max_iters, 0), ('inner_max_iters', inner_max_iters, 1),
        ('preconditioner_probes', preconditioner_probes, 1),
        ('preconditioner_max_size', preconditioner_max_size, 0),
        ('preconditioner_refresh', preconditioner_refresh, 1), ('seed', seed, 0),
    ):
        if not np.isfinite(value) or int(value) != value or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}.')
    if not np.isfinite(tol_grad) or tol_grad < 0:
        raise ValueError('tol_grad must be finite and >= 0.')
    if not np.isfinite(max_radius) or max_radius <= 0:
        raise ValueError('max_radius must be finite and > 0.')
    if initial_radius is not None and (not np.isfinite(initial_radius) or
                                       not 0 < initial_radius <= max_radius):
        raise ValueError('initial_radius must lie in (0, max_radius].')
    if not np.isfinite(acceptance) or not 0 < acceptance < .25:
        raise ValueError('acceptance must lie in (0, .25).')
    if not (np.isfinite(forcing_min) and np.isfinite(forcing_max) and
            0 < forcing_min <= forcing_max < 1):
        raise ValueError('Require 0 < forcing_min <= forcing_max < 1.')
    if not np.isfinite(preconditioner_floor) or not 0 < preconditioner_floor <= 1:
        raise ValueError('preconditioner_floor must lie in (0, 1].')
    if preconditioner not in ('auto', 'linear', 'diagonal', 'identity'):
        raise ValueError('preconditioner must be auto, linear, diagonal or identity.')
    prof = ensure_profiler(profiler)
    recorder = SolverHistoryRecorder(enabled=history, path=history_path,
        line_search_path=trial_history_path, verbose=verbose,
        solver='gauss_newton_krylov', trial_kind='trust_region')
    settings = dict(max_iters=int(max_iters), tol_grad=tol_grad, initial_radius=initial_radius,
        max_radius=max_radius, acceptance=acceptance, inner_max_iters=int(inner_max_iters),
        forcing_min=forcing_min, forcing_max=forcing_max, preconditioner=preconditioner,
        preconditioner_probes=int(preconditioner_probes), preconditioner_max_size=int(preconditioner_max_size),
        preconditioner_floor=preconditioner_floor, preconditioner_refresh=int(preconditioner_refresh), seed=int(seed))
    with prof.span('solve.setup'):
        model = OperatorLinearizationProblem(problem, weighted=weighted,
            term_indices=None if term_indices is None else tuple(term_indices))
        if x0 is not None:
            model.set_point(as_vec(x0, expected_size=model.n_total, name='x0'))
        req = model.required_list(None if required is None else tuple(required))
        start = np.asarray(model.get_point(), dtype=float).copy()
        r, J = model.linearize(required=req)
        g = J.T @ r
    x = start.copy()
    cost = initial_cost = float(r @ r)
    if not np.isfinite(cost):
        raise ValueError('Initial objective must be finite.')
    initial_gradient = float(np.linalg.norm(g))
    radius = initial_radius
    step_norm = 0.
    iterations = accepted_steps = 0
    inner_solves = []
    rng = np.random.default_rng(int(seed))
    inverse = None
    kind = 'unused'
    built_at = -1
    accepts_gradient = False
    if on_iter is not None:
        try:
            inspect.signature(on_iter).bind(0, 0., 0., g.copy())
            accepts_gradient = True
        except (TypeError, ValueError):
            pass

    def stationary(gradient):
        return float(np.max(np.abs(gradient), initial=0.)) <= tol_grad

    def fields():
        result = dict(objective=cost, residual_norm=float(np.linalg.norm(r)),
                      jt_r_inf_norm=float(np.max(np.abs(g), initial=0.)))
        if history_vectors:
            result.update(variables=x.tolist(), jt_r=g.tolist())
        return result

    recorder.emit('initial', 0, **fields(), step_norm=None, trust_radius=radius,
        objective_definition='squared_residual_norm', variable_space='solver', settings=settings)
    status, reason = 'max_iters', 'max_iters'
    for iterations in range(1, int(max_iters)+1):
        if stationary(g):
            iterations -= 1
            status, reason = 'converged', 'gradient_tolerance'
            break
        if on_iter is not None:
            args = (iterations-1, float(np.linalg.norm(r)), step_norm)
            on_iter(*args, g.copy()) if accepts_gradient else on_iter(*args)
        if inverse is None or (kind != 'linear' and built_at != accepted_steps and
                               accepted_steps % int(preconditioner_refresh) == 0):
            with prof.span('solve.preconditioner'):
                inverse, kind = _preconditioner(model.model, J, preconditioner,
                    probes=int(preconditioner_probes), max_size=int(preconditioner_max_size),
                    floor=preconditioner_floor, rng=rng)
            built_at = accepted_steps
        if radius is None:
            # A cheap preconditioned-gradient scale, not a Jacobian norm scan.
            radius = min(max_radius, max(1., 2.*float(np.linalg.norm(inverse(g)))))
        eta = max(forcing_min, forcing_max*min(1., np.sqrt(np.linalg.norm(g)/initial_gradient)))
        with prof.span('solve.iter.step'):
            h, Jh, info = _truncated_cg(J, g, radius, inverse,
                tolerance=eta, max_iters=int(inner_max_iters))
        info.update(tolerance=eta, preconditioner=kind)
        inner_solves.append(info)
        recorder.emit('linear_solve', iterations, **info, trust_radius=radius)
        trial_x = x+h
        displacement = trial_x-x
        if np.array_equal(trial_x, x):
            status, reason = 'stalled', 'unrepresentable_step'
            break
        if not np.array_equal(displacement, h):
            Jh = J @ displacement
        predicted = float(-2.*(g @ displacement)-Jh @ Jh)
        base_cost = cost
        previous_radius = radius
        accepted = False
        rho = float('-inf')
        trial_cost = float('inf')
        trial_reason = 'nonpositive_prediction'
        try:
            if np.isfinite(predicted) and predicted > 0 and np.all(np.isfinite(trial_x)):
                model.set_point(trial_x)
                with prof.span('solve.iter.trial'):
                    trial_r = np.asarray(model.eval(required=req), dtype=float).reshape(-1)
                if trial_r.shape != r.shape:
                    raise ValueError('Residual dimension changed at trial point.')
                trial_cost = float(trial_r @ trial_r)
                reduction = cost-trial_cost
                rho = reduction/predicted
                noise = 8.*np.finfo(float).eps*max(abs(cost), abs(trial_cost))
                accepted = bool(np.isfinite(trial_cost) and reduction > noise and rho > acceptance)
                trial_reason = 'accepted' if accepted else 'insufficient_decrease'
                checked = None
                if np.isfinite(trial_cost) and (accepted or abs(reduction) <= noise):
                    with prof.span('solve.iter.linearize'):
                        new_r, new_J = model.linearize(required=req)
                        new_g = new_J.T @ new_r
                    checked = (new_r, new_J, new_g)
                    if abs(reduction) <= noise:
                        accepted = bool(trial_cost <= cost+noise and
                            (stationary(new_g) or np.max(np.abs(new_g), initial=0.) <=
                             .5*np.max(np.abs(g), initial=0.)))
                        trial_reason = 'gradient_progress' if accepted else 'roundoff_no_progress'
                if accepted:
                    x = trial_x.copy()
                    r, J, g = checked
                    cost = float(r @ r)
                    accepted_steps += 1
            else:
                trial_reason = 'invalid_prediction'
        except Exception as error:
            recorder.emit('trust_region_trial', iterations, accepted=False, reason='evaluation_error',
                error=str(error), trust_radius=previous_radius, objective=None, trial=iterations)
            raise
        finally:
            model.set_point(x)
        boundary = info['status'] in ('boundary', 'zero_curvature')
        if not accepted or rho < .25:
            radius *= .25
        elif rho > .75 and boundary:
            radius = min(max_radius, 2.*radius)
        step_norm = float(np.linalg.norm(displacement)) if accepted else 0.
        recorder.emit('trust_region_trial', iterations, objective=trial_cost, base_objective=base_cost,
            predicted_reduction=predicted, actual_reduction=base_cost-trial_cost,
            gain_ratio=rho, trust_radius=previous_radius, next_trust_radius=radius,
            accepted=accepted, reason=trial_reason, step_norm=float(np.linalg.norm(displacement)), trial=iterations)
        recorder.emit('iteration_end', iterations, **fields(), step_norm=step_norm,
            trust_radius=radius, accepted=accepted, reason=trial_reason, gain_ratio=rho)
        if not stationary(g) and radius <= np.finfo(float).eps*max(1., np.linalg.norm(x)):
            status, reason = 'stalled', 'trust_radius_too_small'
            break
    if stationary(g):
        status, reason = 'converged', 'gradient_tolerance'
    recorder.emit('final', iterations, **fields(), step_norm=step_norm, trust_radius=radius,
                  status=status, reason=reason, gradient_converged=stationary(g))
    return SolveOutcome(solution=x, stats=SolveStats(status=status, iterations=iterations,
        initial_objective=initial_cost, objective=cost, residual_norm=float(np.linalg.norm(r)),
        step_norm=step_norm, message=reason), timing=prof.snapshot(),
        meta=dict(solver='gauss_newton_krylov', x0=start, settings=settings, reason=reason,
                  accepted_steps=accepted_steps, gradient_converged=stationary(g),
                  preconditioner=kind, inner_solves=inner_solves),
        history=recorder.events, trial_history=recorder.line_search_events)


def solve_gauss_newton_krylov(
    problem: Any, max_iters: int = 200, *, x0: Any = None,
    required: Iterable[StateKey] | None = None, weighted: bool | None = None,
    term_indices: Iterable[int] | None = None,
    tol_r: float = 1e-10, tol_dx: float = 1e-12, tol_grad: float = 1e-10,
    damping: float = 1e-8, inner_tol: float = 1e-10,
    inner_max_iters: int | None = None, line_search: bool = True,
    ls_beta: float = .5, ls_min_step: float = 1e-8, ls_max_iters: int = 12,
    ls_max_retries: int = 3, damping_increase: float = 10., damping_max: float = 1e12,
    damping_decrease: float = .1, damping_min_factor: float = 100.,
    c_armijo: float = 1e-4, globalization: str = 'line_search',
    preconditioner: str = 'auto', preconditioner_probes: int = 8,
    preconditioner_max_size: int = 512, preconditioner_floor: float = 1e-10,
    seed: int = 0,
    # Legacy trust-region controls are used only with globalization="trust_region".
    initial_radius: float | None = None, max_radius: float = 1e8,
    acceptance: float = .1, forcing_min: float = 1e-4, forcing_max: float = .1,
    preconditioner_refresh: int = 5,
    history: bool = True, history_vectors: bool = False,
    history_path: str | Path | None = None,
    line_search_history_path: str | Path | None = None,
    trial_history_path: str | Path | None = None,
    verbose: bool = True, on_iter: Callable[..., None] | None = None,
    profiler: Profiler | None = None,
) -> SolveOutcome:
    """Gauss-Newton globalization with a diagonally scaled Krylov CGLS step.

    The default uses the same damping, Armijo search, retry policy and
    convergence test as ``gauss_newton``. Only the linear least-squares step
    differs: it uses JVP/VJP products and a probe-estimated right scaling.
    ``globalization='trust_region'`` retains the previous PCG solver.
    """
    if globalization == 'trust_region':
        return _solve_gauss_newton_krylov_trust_region(
            problem, max_iters=max_iters, x0=x0, required=required, weighted=weighted,
            term_indices=term_indices, tol_grad=tol_grad, initial_radius=initial_radius,
            max_radius=max_radius, acceptance=acceptance,
            inner_max_iters=50 if inner_max_iters is None else inner_max_iters,
            forcing_min=forcing_min, forcing_max=forcing_max,
            preconditioner=preconditioner, preconditioner_probes=preconditioner_probes,
            preconditioner_max_size=preconditioner_max_size,
            preconditioner_floor=preconditioner_floor,
            preconditioner_refresh=preconditioner_refresh, seed=seed,
            history=history, history_vectors=history_vectors, history_path=history_path,
            trial_history_path=trial_history_path or line_search_history_path,
            verbose=verbose, on_iter=on_iter, profiler=profiler)
    if globalization != 'line_search':
        raise ValueError("globalization must be 'line_search' or 'trust_region'.")
    if preconditioner not in ('auto', 'linear', 'diagonal', 'identity'):
        raise ValueError('preconditioner must be auto, linear, diagonal or identity.')
    if not np.isfinite(preconditioner_probes) or int(preconditioner_probes) != preconditioner_probes or preconditioner_probes <= 0:
        raise ValueError('preconditioner_probes must be a positive integer.')
    if not np.isfinite(preconditioner_floor) or not 0 < preconditioner_floor <= 1:
        raise ValueError('preconditioner_floor must lie in (0, 1].')
    if not np.isfinite(preconditioner_max_size) or int(preconditioner_max_size) != preconditioner_max_size or preconditioner_max_size < 0:
        raise ValueError('preconditioner_max_size must be a nonnegative integer.')
    if not np.isfinite(seed) or int(seed) != seed or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    if not np.isfinite(preconditioner_refresh) or int(preconditioner_refresh) != preconditioner_refresh or preconditioner_refresh < 1:
        raise ValueError('preconditioner_refresh must be a positive integer.')
    if globalization == 'line_search':
        if initial_radius is not None or max_radius != 1e8 or acceptance != .1 or \
                forcing_min != 1e-4 or forcing_max != .1:
            raise ValueError('Trust-region options require globalization="trust_region".')

    rng = np.random.default_rng(int(seed))
    cache: dict[int, tuple[Any, np.ndarray, float, str]] = {}
    shared_estimate: tuple[np.ndarray, float, str] | None = None
    estimated_linearizations = 0

    def estimate(J):
        nonlocal shared_estimate, estimated_linearizations
        key = id(J)
        if key in cache and cache[key][0] is J:
            return cache[key][1:]
        cache.clear()
        should_refresh = shared_estimate is None or (
            shared_estimate[2] not in ('linear_diagonal', 'identity')
            and estimated_linearizations % int(preconditioner_refresh) == 0
        )
        estimated_linearizations += 1
        if not should_refresh:
            cache[key] = (J, *shared_estimate)
            return shared_estimate
        n = J.shape[1]
        diagonal = None
        kind = 'identity'
        if preconditioner in ('auto', 'linear') and n <= preconditioner_max_size:
            provider = getattr(J.model, 'linear_residual_gram', None)
            gram = provider(max_size=int(preconditioner_max_size)) if callable(provider) else None
            if gram is not None:
                gram = np.asarray(gram, dtype=float)
                if gram.shape != (n, n) or not np.all(np.isfinite(gram)):
                    raise ValueError('Linear residual Gram must be finite and n_total by n_total.')
                diagonal = np.maximum(np.diag(gram), 0.)
                kind = 'linear_diagonal'
            elif preconditioner == 'linear':
                raise ValueError('No affine-residual preconditioner within the size budget.')
        if diagonal is None and preconditioner != 'identity':
            diagonal = np.zeros(n)
            for _ in range(int(preconditioner_probes)):
                w = rng.choice(np.array([-1., 1.]), size=J.shape[0])
                product = J.T @ w
                diagonal += product * product / int(preconditioner_probes)
            kind = 'diagonal'
        if diagonal is None or not np.any(diagonal):
            scale = np.ones(n)
            max_diagonal = 0.
        else:
            max_diagonal = float(np.max(diagonal, initial=0.))
            if not np.isfinite(max_diagonal):
                raise ValueError('Preconditioner estimate overflow; rescale the problem.')
            relative_diagonal = np.maximum(diagonal / max_diagonal, preconditioner_floor)
            scale = 1. / np.sqrt(relative_diagonal)
        floor = float(damping_min_factor * np.finfo(float).eps * max_diagonal)
        shared_estimate = (scale, floor, kind)
        cache[key] = (J, *shared_estimate)
        return shared_estimate

    def step_solver(r, J, damp):
        scale, _, kind = estimate(J)
        dx, info = cgls_step(J, r, damp, tolerance=inner_tol,
                             max_iters=inner_max_iters, coordinate_scale=scale)
        info['preconditioner'] = kind
        return dx, info

    def damping_floor(J):
        return estimate(J)[1]

    return solve_gauss_newton_operator(
        problem, max_iters=max_iters, x0=x0, required=required, weighted=weighted,
        term_indices=term_indices, tol_r=tol_r, tol_dx=tol_dx, tol_grad=tol_grad,
        damping=damping, inner_tol=inner_tol, inner_max_iters=inner_max_iters,
        line_search=line_search, ls_beta=ls_beta, ls_min_step=ls_min_step,
        ls_max_iters=ls_max_iters, ls_max_retries=ls_max_retries,
        damping_increase=damping_increase, damping_max=damping_max,
        damping_decrease=damping_decrease, damping_min_factor=damping_min_factor,
        c_armijo=c_armijo, history=history, history_vectors=history_vectors,
        history_path=history_path,
        line_search_history_path=line_search_history_path or trial_history_path,
        verbose=verbose, on_iter=on_iter, profiler=profiler,
        _step_solver=step_solver, _damping_floor_fn=damping_floor,
        _solver_name='gauss_newton_krylov')


__all__ = ['solve_gauss_newton_krylov']
