"""liteopt solver integration with lazy optional dependency loading."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
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
    _format_option_names,
)


def _parse_liteopt_gd_result(
    result: Any,
    *,
    n_total: int,
) -> tuple[Array, bool, int | None]:
    if isinstance(result, Mapping):
        x_raw = result.get("x_star", result.get("x", None))
        if x_raw is None:
            raise ValueError("solve_liteopt_gd: liteopt.gd result must provide x_star (or x).")
        converged = bool(result.get("converged", result.get("success", False)))
        iters_raw = result.get("iters", result.get("n_iters", result.get("nit", None)))
        iters = None if iters_raw is None else int(iters_raw)
        x_star = as_vec(x_raw, expected_size=n_total, name="result.x")
        return x_star, converged, iters

    if isinstance(result, (tuple, list)):
        if len(result) < 3:
            raise ValueError(
                "solve_liteopt_gd: liteopt.gd result must be "
                "(x_star, f_star, converged) or Mapping."
            )
        x_star = as_vec(result[0], expected_size=n_total, name="result.x")
        converged = bool(result[2])
        iters = None
        if len(result) >= 4:
            iters = int(result[3])
        return x_star, converged, iters

    x_star = as_vec(result, expected_size=n_total, name="result.x")
    return x_star, False, None


def _parse_liteopt_gn_result(
    result: Any,
    *,
    n_total: int,
) -> tuple[Array, float, int | None, float, float, bool]:
    if isinstance(result, Mapping):
        x_raw = result.get("x_star", result.get("x", None))
        if x_raw is None:
            raise ValueError("solve_liteopt_gd: liteopt.gn result must provide x_star (or x).")
        cost_raw = result.get("cost", result.get("f_star", float("nan")))
        iters_raw = result.get("iters", result.get("n_iters", result.get("nit", None)))
        rnorm_raw = result.get("r_norm", result.get("residual_norm", float("nan")))
        dxnorm_raw = result.get("dx_norm", result.get("step_norm", float("nan")))
        ok_raw = result.get("ok", result.get("converged", result.get("success", False)))
        x_star = as_vec(x_raw, expected_size=n_total, name="result.x")
        iters = None if iters_raw is None else int(iters_raw)
        return (
            x_star,
            float(cost_raw),
            iters,
            float(rnorm_raw),
            float(dxnorm_raw),
            bool(ok_raw),
        )

    if isinstance(result, (tuple, list)):
        if len(result) < 6:
            raise ValueError(
                "solve_liteopt_gd: liteopt.gn result must be "
                "(x_star, cost, iters, r_norm, dx_norm, ok) or Mapping."
            )
        x_star = as_vec(result[0], expected_size=n_total, name="result.x")
        return (
            x_star,
            float(result[1]),
            int(result[2]),
            float(result[3]),
            float(result[4]),
            bool(result[5]),
        )

    raise ValueError(
        "solve_liteopt_gd: unsupported liteopt.gn result type. "
        "Expected Mapping, tuple, or list."
    )


def _liteopt_uses_options_api(fn: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    return "options" in signature.parameters


def _silence_liteopt_output(fn: Callable[..., Any], options: dict[str, Any]) -> None:
    """Reserve top-level verbose for Rei while honoring explicit backend options."""
    if "verbose" in options:
        return
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return
    if "verbose" in parameters or "options" in parameters:
        options["verbose"] = False


def _call_liteopt_gd(
    liteopt: Any,
    fun: Callable[[Array], float],
    grad: Callable[[Array], Array],
    x0: Array,
    options: Mapping[str, Any],
) -> Any:
    gd = liteopt.gd
    if _liteopt_uses_options_api(gd):
        return gd(fun, grad, x0, options=dict(options))
    return gd(fun, grad, x0, **dict(options))


def _call_liteopt_gn(
    liteopt: Any,
    residual: Callable[[Array], Array],
    jacobian: Callable[[Array], Array],
    x0: Array,
    options: Mapping[str, Any],
) -> Any:
    gn = liteopt.gn
    if _liteopt_uses_options_api(gn):
        return gn(residual, x0, jacobian=jacobian, options=dict(options))
    return gn(residual, jacobian, x0, **dict(options))


def solve_liteopt_gd(
    problem: Any,
    *,
    x0: Array | Any = None,
    required: Iterable[StateKey] | None = None,
    weighted: bool | None = None,
    term_indices: Iterable[int] | None = None,
    method: str = "gd",
    verbose: bool | None = True,
    max_iters: int | None = 200,
    step_size: float = 1e-3,
    tol_grad: float = 1e-4,
    tol_r: float | None = None,
    tol_dx: float | None = None,
    lambda_: float | None = None,
    step_scale: float | None = None,
    damping_update: str | None = None,
    linear_system: str | None = None,
    line_search_method: str | None = None,
    line_search: bool | None = None,
    ls_beta: float | None = None,
    ls_min_step: float | None = None,
    ls_max_steps: int | None = None,
    ls_max_iters: int | None = None,
    c_armijo: float | None = None,
    nonfinite_retries: int = 8,
    nonfinite_step_shrink: float = 0.2,
    min_step_size: float = 1e-12,
    options: Mapping[str, Any] | None = None,
    on_iter: IterCallback | None = None,
    profiler: Profiler | None = None,
) -> SolveResult:
    """Solve `||r(x)||^2` via liteopt (`gd` or `gn`) and return SolveOutcome."""

    prof = ensure_profiler(profiler)
    try:
        import liteopt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "solve_liteopt_gd requires liteopt. Install liteopt and re-run."
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
    method_key = str(method).strip().lower()
    if method_key not in {"gd", "gn"}:
        raise ValueError(
            "solve_liteopt_gd: liteopt method must be 'gd' or 'gn'. "
            f"Got method={method!r}."
        )
    progress = BackendProgress(f"liteopt_{method_key}", verbose=bool(verbose))
    progress.initial(initial_cost, grad0 / 2)

    iter_count = 0
    last_dxnorm = 0.0
    prev_x = x0_init.copy()

    def _is_nonfinite_error(exc: BaseException) -> bool:
        if isinstance(exc, FloatingPointError):
            return True
        msg = str(exc).lower()
        return (
            "non-finite" in msg
            or "must return finite" in msg
            or "nan" in msg
            or "inf" in msg
            or "overflow" in msg
        )

    if method_key == "gn":
        options_local: dict[str, Any] = {} if options is None else dict(options)
        _silence_liteopt_output(liteopt.gn, options_local)
        if max_iters is not None and "max_iters" not in options_local:
            options_local["max_iters"] = int(max_iters)
        if tol_r is not None and "tol_r" not in options_local:
            options_local["tol_r"] = float(tol_r)
        if tol_dx is not None and "tol_dx" not in options_local:
            options_local["tol_dx"] = float(tol_dx)
        if lambda_ is not None and "lambda_" not in options_local:
            options_local["lambda_"] = float(lambda_)
        if step_scale is not None and "step_scale" not in options_local:
            options_local["step_scale"] = float(step_scale)
        if damping_update is not None and "damping_update" not in options_local:
            options_local["damping_update"] = str(damping_update)
        if linear_system is not None and "linear_system" not in options_local:
            options_local["linear_system"] = str(linear_system)
        if line_search_method is not None and "line_search_method" not in options_local:
            options_local["line_search_method"] = str(line_search_method)
        if line_search is not None and "line_search" not in options_local:
            options_local["line_search"] = bool(line_search)
        if ls_beta is not None and "ls_beta" not in options_local:
            options_local["ls_beta"] = float(ls_beta)
        if ls_min_step is not None and "ls_min_step" not in options_local:
            options_local["ls_min_step"] = float(ls_min_step)
        if "ls_max_steps" not in options_local:
            if ls_max_steps is not None:
                options_local["ls_max_steps"] = int(ls_max_steps)
            elif ls_max_iters is not None:
                options_local["ls_max_steps"] = int(ls_max_iters)
        if c_armijo is not None and "c_armijo" not in options_local:
            options_local["c_armijo"] = float(c_armijo)

        gn_iter_count = 0
        gn_prev_x: Array | None = None

        def _emit_gn_iter(x_vec: Array, r_vec: Array) -> None:
            nonlocal gn_iter_count, gn_prev_x, last_dxnorm
            if on_iter is None:
                return
            if gn_prev_x is not None and np.array_equal(gn_prev_x, x_vec):
                return
            if gn_prev_x is None:
                dxnorm_local = 0.0
            else:
                dxnorm_local = float(np.linalg.norm(x_vec - gn_prev_x))
            last_dxnorm = dxnorm_local
            on_iter(gn_iter_count, float(np.linalg.norm(r_vec)), dxnorm_local)
            gn_iter_count += 1
            gn_prev_x = x_vec.copy()

        def residual(x: Array) -> Array:
            x_vec = as_vec(x, expected_size=n_total, name="x")
            if not bool(np.all(np.isfinite(x_vec))):
                raise ValueError("solve_liteopt_gd: non-finite iterate encountered.")
            linear_problem.set_point(x_vec)
            r_all, _J_all = linear_problem.linearize(required=req)
            r = np.asarray(r_all, dtype=float).reshape(-1)
            if not bool(np.all(np.isfinite(r))):
                raise ValueError("solve_liteopt_gd: residual became non-finite.")
            _emit_gn_iter(x_vec, r)
            return r

        def jacobian(x: Array) -> Array:
            x_vec = as_vec(x, expected_size=n_total, name="x")
            if not bool(np.all(np.isfinite(x_vec))):
                raise ValueError("solve_liteopt_gd: non-finite iterate encountered.")
            linear_problem.set_point(x_vec)
            _r_all, J_all = linear_problem.linearize(required=req)
            J = np.asarray(J_all, dtype=float)
            if not bool(np.all(np.isfinite(J))):
                raise ValueError("solve_liteopt_gd: jacobian became non-finite.")
            return J

        result: Any | None = None
        failure_message = ""
        with prof.span("solve.backend"):
            try:
                result = _call_liteopt_gn(
                    liteopt,
                    residual,
                    jacobian,
                    x0_init.copy(),
                    options_local,
                )
            except Exception as e:
                if not _is_nonfinite_error(e):
                    raise
                failure_message = str(e)
                result = None

        converged = False
        x_star = x0_init.copy()
        iters = int(max_iters) if max_iters is not None else 0
        cost = float("nan")
        rnorm = float("nan")
        if result is not None:
            (
                x_star,
                cost,
                iters_raw,
                rnorm,
                last_dxnorm,
                converged,
            ) = _parse_liteopt_gn_result(result, n_total=n_total)
            if iters_raw is not None:
                iters = int(iters_raw)
        with prof.span("solve.finalize"):
            final_gradient = None
            if bool(np.all(np.isfinite(x_star))):
                cost_eval, final_gradient, rnorm_eval = objective.eval(x_star)
                if bool(np.isfinite(cost_eval)):
                    cost = float(cost_eval)
                if bool(np.isfinite(rnorm_eval)):
                    rnorm = float(rnorm_eval)
            else:
                failure_message = "solve_liteopt_gd: backend returned a non-finite solution vector."

        finite_objective = bool(np.isfinite(cost)) and bool(np.isfinite(rnorm))
        status = "converged" if bool(converged and finite_objective) else "failed"
        message = ""
        if result is None:
            message = "liteopt.gn aborted due to non-finite objective/residual."
            if str(failure_message).strip():
                message += f" last_error={failure_message}"
        elif not finite_objective:
            message = (
                "liteopt.gn returned a non-finite final objective/residual; "
                "marking solve as failed."
            )
        outcome = SolveOutcome(
            solution=x_star.copy(),
            stats=SolveStats(
                status=status,
                iterations=int(iters),
                initial_objective=float(initial_cost),
                objective=float(cost),
                residual_norm=float(rnorm),
                step_norm=float(last_dxnorm),
                message=message,
            ),
            timing=prof.snapshot(),
            meta={
                "solver": "liteopt_gn",
                "method": "gn",
                "x0": x0_init.copy(),
            },
        )
        progress.final(outcome, None if final_gradient is None else final_gradient / 2)
        return outcome

    options_local = {} if options is None else dict(options)
    _silence_liteopt_output(liteopt.gd, options_local)
    nonfinite_retries = int(options_local.pop("nonfinite_retries", nonfinite_retries))
    nonfinite_step_shrink = float(
        options_local.pop("nonfinite_step_shrink", nonfinite_step_shrink)
    )
    min_step_size = float(options_local.pop("min_step_size", min_step_size))

    if nonfinite_retries < 0:
        raise ValueError(
            "solve_liteopt_gd: nonfinite_retries must be >= 0, "
            f"got {nonfinite_retries}."
        )
    if not (0.0 < nonfinite_step_shrink < 1.0):
        raise ValueError(
            "solve_liteopt_gd: nonfinite_step_shrink must be in (0, 1), "
            f"got {nonfinite_step_shrink}."
        )
    if min_step_size <= 0.0:
        raise ValueError(
            "solve_liteopt_gd: min_step_size must be > 0, "
            f"got {min_step_size}."
        )

    gn_only_keys = {
        "tol_r",
        "tol_dx",
        "lambda_",
        "step_scale",
        "damping_update",
        "linear_system",
        "line_search_method",
        "ls_beta",
        "ls_min_step",
        "ls_max_steps",
        "ls_max_iters",
        "c_armijo",
    }
    bad_for_gd = tuple(sorted(k for k in options_local if str(k) in gn_only_keys))
    if bad_for_gd:
        raise ValueError(
            "solve_liteopt_gd(method='gd'): unsupported option(s) for gd: "
            f"{_format_option_names(bad_for_gd)}. "
            "Use method='gn' to enable these options."
        )

    if "step_size" not in options_local:
        options_local["step_size"] = float(step_size)
    if max_iters is not None and "max_iters" not in options_local:
        options_local["max_iters"] = int(max_iters)
    if "tol_grad" not in options_local:
        options_local["tol_grad"] = float(tol_grad)
    if line_search is not None and "line_search" not in options_local:
        options_local["line_search"] = bool(line_search)
    base_step_size = float(options_local.get("step_size", step_size))
    if base_step_size <= 0.0:
        raise ValueError(
            "solve_liteopt_gd: step_size must be > 0, "
            f"got {base_step_size}."
        )

    def fun(x: Array) -> float:
        x_vec = as_vec(x, expected_size=n_total, name="x")
        if not bool(np.all(np.isfinite(x_vec))):
            raise ValueError("solve_liteopt_gd: non-finite iterate encountered.")
        fx, _gx, _rnorm = objective.eval(x_vec)
        if not bool(np.isfinite(fx)):
            raise ValueError("solve_liteopt_gd: objective became non-finite.")
        return float(fx)

    def grad(x: Array) -> Array:
        nonlocal iter_count, last_dxnorm, prev_x
        x_vec = as_vec(x, expected_size=n_total, name="x")
        if not bool(np.all(np.isfinite(x_vec))):
            raise ValueError("solve_liteopt_gd: non-finite iterate encountered.")
        fx, gx, rnorm = objective.eval(x_vec)
        if not bool(np.isfinite(rnorm)):
            raise ValueError("solve_liteopt_gd: residual norm became non-finite.")
        if not bool(np.all(np.isfinite(gx))):
            raise ValueError("solve_liteopt_gd: gradient became non-finite.")
        last_dxnorm = float(np.linalg.norm(x_vec - prev_x))
        if last_dxnorm > 0:
            progress.iterate(fx, last_dxnorm, gx / 2)
        prev_x = x_vec.copy()
        if on_iter is not None:
            on_iter(iter_count, rnorm, last_dxnorm)
        iter_count += 1
        return gx

    result: Any | None = None
    step_size_used = base_step_size
    retry_count = 0
    failure_message = ""

    with prof.span("solve.backend"):
        for attempt in range(nonfinite_retries + 1):
            options_attempt = dict(options_local)
            options_attempt["step_size"] = float(step_size_used)
            prev_x = x0_init.copy()
            try:
                result = _call_liteopt_gd(
                    liteopt,
                    fun,
                    grad,
                    x0_init.copy(),
                    options_attempt,
                )
                break
            except Exception as e:
                if not _is_nonfinite_error(e):
                    raise
                failure_message = str(e)
                retry_count = int(attempt + 1)
                step_next = float(step_size_used * nonfinite_step_shrink)
                if attempt >= nonfinite_retries or step_next < min_step_size:
                    result = None
                    break
                step_size_used = step_next

    converged = False
    iters_from_result: int | None = None
    x_star = x0_init.copy()
    if result is not None:
        x_star, converged, iters_from_result = _parse_liteopt_gd_result(
            result,
            n_total=n_total,
        )

    cost = float("nan")
    rnorm = float("nan")
    final_gradient = None
    with prof.span("solve.finalize"):
        if bool(np.all(np.isfinite(x_star))):
            cost, final_gradient, rnorm = objective.eval(x_star)
        else:
            failure_message = (
                "solve_liteopt_gd: backend returned a non-finite solution vector."
            )

    iters = int(iter_count)
    if iters_from_result is not None and retry_count <= 0:
        iters = int(iters_from_result)
    elif iters <= 0:
        iters = int(max_iters) if max_iters is not None else 0

    if iter_count <= 0:
        last_dxnorm = float(np.linalg.norm(x_star - x0_init))

    finite_objective = bool(np.isfinite(cost)) and bool(np.isfinite(rnorm))
    status = "converged" if bool(converged and finite_objective) else "failed"
    message = ""
    if result is None:
        message = (
            "liteopt.gd aborted due to non-finite objective/gradient. "
            f"retries={retry_count}, final_step_size={step_size_used:.3e}."
        )
        if str(failure_message).strip():
            message += f" last_error={failure_message}"
    elif not finite_objective:
        message = (
            "liteopt.gd returned a non-finite final objective/residual; "
            "marking solve as failed."
        )

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
            "solver": "liteopt_gd",
            "method": "gd",
            "retry_count": int(retry_count),
            "step_size_used": float(step_size_used),
            "x0": x0_init.copy(),
        },
    )
    progress.final(outcome, None if final_gradient is None else final_gradient / 2)
    return outcome


__all__ = ["solve_liteopt_gd"]
