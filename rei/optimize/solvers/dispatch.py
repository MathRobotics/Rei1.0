from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from ...core.outcome import SolveOutcome, SolveStats
from ...core.state_cache import StateKey
from ...core.timing import Profiler, ensure_profiler
from ...problem import LinearizedProblem, as_linearized_problem
from ...xops import as_vec
from .gauss_newton import solve_gauss_newton
from .nls import nls

Array = np.ndarray
SolveResult = SolveOutcome
IterCallback = Callable[[int, float, float], None]

_COMMON_SOLVE_OPTION_KEYS = frozenset(
    {
        "x0",
        "weighted",
        "term_indices",
    }
)

_SOLVER_REI_OPTION_KEYS: dict[str, frozenset[str]] = {
    "gauss_newton": frozenset(
        {
            "max_iters",
            "tol_r",
            "tol_dx",
            "damping",
            "line_search",
            "ls_beta",
            "ls_min_step",
            "ls_max_iters",
        }
    ),
    "scipy_minimize": frozenset(
        {
            "method",
            "max_iters",
            "tol",
            "bounds",
        }
    ),
    "cyipopt": frozenset(
        {
            "max_iters",
            "tol",
            "bounds",
        }
    ),
    "liteopt": frozenset(
        {
            "method",
            "verbose",
            "max_iters",
            "step_size",
            "tol_grad",
            "tol_r",
            "tol_dx",
            "lambda_",
            "step_scale",
            "damping_update",
            "linear_system",
            "line_search_method",
            "line_search",
            "ls_beta",
            "ls_min_step",
            "ls_max_steps",
            "ls_max_iters",
            "c_armijo",
            "nonfinite_retries",
            "nonfinite_step_shrink",
            "min_step_size",
        }
    ),
}

_ALL_REI_SOLVER_OPTION_KEYS = frozenset().union(*_SOLVER_REI_OPTION_KEYS.values())


def _allowed_solve_option_keys(solver_key: str) -> frozenset[str]:
    allowed = set(_COMMON_SOLVE_OPTION_KEYS)
    allowed.update(_SOLVER_REI_OPTION_KEYS[solver_key])
    if solver_key != "gauss_newton":
        allowed.add("backend_options")
    return frozenset(allowed)


def _format_option_names(keys: Iterable[str]) -> str:
    return ", ".join(sorted(str(k) for k in keys))


def _normalize_backend_options_for_solver(
    options: Mapping[str, Any],
    *,
    solver_key: str,
) -> Mapping[str, Any] | None:
    allowed = _allowed_solve_option_keys(solver_key)
    unknown = tuple(k for k in options if k not in allowed)
    if not unknown:
        if solver_key == "gauss_newton":
            return None
        return _as_options_mapping(
            options.get("backend_options", None),
            where=f"solve({solver_key})",
        )

    if solver_key == "gauss_newton":
        allowed_text = _format_option_names(_allowed_solve_option_keys("gauss_newton"))
        raise ValueError(
            "solve(gauss_newton): unsupported option(s): "
            f"{_format_option_names(unknown)}. "
            f"Allowed options are: {allowed_text}."
        )

    foreign_rei = tuple(
        k for k in unknown if str(k) in (_ALL_REI_SOLVER_OPTION_KEYS - _SOLVER_REI_OPTION_KEYS[solver_key])
    )
    if foreign_rei:
        allowed_text = _format_option_names(_allowed_solve_option_keys(solver_key))
        raise ValueError(
            f"solve({solver_key}): unsupported solver option(s): "
            f"{_format_option_names(foreign_rei)}. "
            f"Allowed options are: {allowed_text}. "
            "Backend-specific options must be under options['backend_options'] "
            "or additional top-level keys that are not rei solver options."
        )

    backend = _as_options_mapping(
        options.get("backend_options", None),
        where=f"solve({solver_key})",
    )
    backend_local: dict[str, Any] = {} if backend is None else dict(backend)
    for k in unknown:
        if k not in backend_local:
            backend_local[k] = options[k]
    return backend_local


class _LinearizedObjective:
    def __init__(
        self,
        problem: LinearizedProblem,
        *,
        required: Iterable[StateKey] | None = None,
    ) -> None:
        self.problem = problem
        self.required = None if required is None else list(required)
        self._last_x: Array | None = None
        self._last_cost: float = float("nan")
        self._last_grad: Array | None = None
        self._last_rnorm: float = float("inf")

    def eval(self, x: Array) -> tuple[float, Array, float]:
        x_vec = as_vec(x, expected_size=int(self.problem.n_total), name="x")
        if self._last_x is not None and np.array_equal(self._last_x, x_vec):
            if self._last_grad is None:
                raise RuntimeError("internal error: gradient cache is missing.")
            return self._last_cost, self._last_grad, self._last_rnorm

        self.problem.set_point(x_vec)
        r_all, J_all = self.problem.linearize(required=self.required)
        r = np.asarray(r_all, dtype=float).reshape(-1)
        J = np.asarray(J_all, dtype=float)

        cost = float(r @ r)
        grad = np.asarray(2.0 * (J.T @ r), dtype=float).reshape(-1)
        rnorm = float(np.linalg.norm(r))

        self._last_x = x_vec.copy()
        self._last_cost = cost
        self._last_grad = grad
        self._last_rnorm = rnorm
        return cost, grad, rnorm


def _extract_x_from_callback_arg(xk: Any, *, size: int) -> Array | None:
    try:
        return as_vec(xk, expected_size=size, name="callback.x")
    except Exception:
        pass

    x_attr = getattr(xk, "x", None)
    if x_attr is None:
        return None
    try:
        return as_vec(x_attr, expected_size=size, name="callback.x")
    except Exception:
        return None


def _merge_options(*sources: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for src in sources:
        if src is None:
            continue
        merged.update(dict(src))
    return merged


def _as_options_mapping(value: Any, *, where: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{where}: options['backend_options'] must be a mapping or None.")
    return dict(value)

def _normalize_term_indices_option(raw: Any) -> tuple[int, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)):
        raise TypeError("term_indices must be an iterable of integers, not a string.")
    try:
        vals = tuple(int(v) for v in raw)
    except Exception as e:  # pragma: no cover
        raise TypeError("term_indices must be an iterable of integers or None.") from e
    return vals


def _initialize_linear_problem_point(
    linear_problem: LinearizedProblem,
    *,
    x0: Array | Any = None,
) -> Array:
    n_total = int(linear_problem.n_total)
    if x0 is not None:
        linear_problem.set_point(as_vec(x0, expected_size=n_total, name="x0"))
    return np.asarray(linear_problem.get_point(), dtype=float).reshape(-1).copy()


def solve_scipy_minimize(
    problem: Any,
    *,
    x0: Array | Any = None,
    required: Iterable[StateKey] | None = None,
    weighted: bool = True,
    term_indices: Iterable[int] | None = None,
    method: str = "L-BFGS-B",
    max_iters: int | None = 200,
    tol: float | None = None,
    bounds: Any = None,
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
            weighted=bool(weighted),
            term_indices=None if term_indices is None else tuple(int(i) for i in term_indices),
        )
        x0_init = _initialize_linear_problem_point(linear_problem, x0=x0)
        req = linear_problem.required_list(required)
        objective = _LinearizedObjective(linear_problem, required=req)
        n_total = int(x0_init.size)
        initial_cost, _grad0, _rnorm0 = objective.eval(x0_init)

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
        _fx, _gx, rnorm = objective.eval(x_vec)
        last_dxnorm = float(np.linalg.norm(x_vec - prev_x))
        prev_x = x_vec.copy()
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
        cost, _grad, rnorm = objective.eval(x_star)
    iters = int(getattr(result, "nit", iter_count))
    if iters <= 0:
        iters = int(iter_count)
    if iter_count <= 0:
        last_dxnorm = float(np.linalg.norm(x_star - x0_init))
    converged = bool(getattr(result, "success", False))
    status = "converged" if converged else "failed"
    message = str(getattr(result, "message", "") or "")

    return SolveOutcome(
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


def solve_cyipopt_minimize(
    problem: Any,
    *,
    x0: Array | Any = None,
    required: Iterable[StateKey] | None = None,
    weighted: bool = True,
    term_indices: Iterable[int] | None = None,
    max_iters: int | None = 200,
    tol: float | None = None,
    bounds: Any = None,
    options: Mapping[str, Any] | None = None,
    on_iter: IterCallback | None = None,
    profiler: Profiler | None = None,
) -> SolveResult:
    """Solve `||r(x)||^2` via cyipopt.minimize_ipopt and return SolveOutcome."""

    prof = ensure_profiler(profiler)
    try:
        from cyipopt import minimize_ipopt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "solve_cyipopt_minimize requires cyipopt. Install cyipopt and re-run."
        ) from e

    with prof.span("solve.setup"):
        linear_problem = as_linearized_problem(
            problem,
            weighted=bool(weighted),
            term_indices=None if term_indices is None else tuple(int(i) for i in term_indices),
        )
        x0_init = _initialize_linear_problem_point(linear_problem, x0=x0)
        req = linear_problem.required_list(required)
        objective = _LinearizedObjective(linear_problem, required=req)
        n_total = int(x0_init.size)
        initial_cost, _grad0, _rnorm0 = objective.eval(x0_init)

    options_local: dict[str, Any] = {} if options is None else dict(options)
    if max_iters is not None and "max_iter" not in options_local:
        options_local["max_iter"] = int(max_iters)

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
        _fx, _gx, rnorm = objective.eval(x_vec)
        last_dxnorm = float(np.linalg.norm(x_vec - prev_x))
        prev_x = x_vec.copy()
        if on_iter is not None:
            on_iter(iter_count, rnorm, last_dxnorm)
        iter_count += 1

    kwargs: dict[str, Any] = {
        "fun": fun,
        "x0": x0_init,
        "jac": jac,
        "bounds": bounds,
        "options": options_local,
    }
    if tol is not None:
        kwargs["tol"] = float(tol)
    if on_iter is not None:
        kwargs["callback"] = callback

    with prof.span("solve.backend"):
        try:
            result = minimize_ipopt(**kwargs)
        except TypeError as e:
            if "callback" not in str(e):
                raise
            kwargs.pop("callback", None)
            result = minimize_ipopt(**kwargs)

    with prof.span("solve.finalize"):
        x_star = as_vec(getattr(result, "x", x0_init), expected_size=n_total, name="result.x")
        cost, _grad, rnorm = objective.eval(x_star)
    iters = int(getattr(result, "nit", iter_count))
    if iters <= 0:
        iters = int(iter_count)
    if iter_count <= 0:
        last_dxnorm = float(np.linalg.norm(x_star - x0_init))
    converged = bool(getattr(result, "success", False))
    status = "converged" if converged else "failed"
    message = str(getattr(result, "message", "") or "")

    return SolveOutcome(
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
            "solver": "cyipopt_minimize",
            "x0": x0_init.copy(),
        },
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
    weighted: bool = True,
    term_indices: Iterable[int] | None = None,
    method: str = "gd",
    verbose: bool | None = None,
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
            weighted=bool(weighted),
            term_indices=None if term_indices is None else tuple(int(i) for i in term_indices),
        )
        x0_init = _initialize_linear_problem_point(linear_problem, x0=x0)
        req = linear_problem.required_list(required)
        objective = _LinearizedObjective(linear_problem, required=req)
        n_total = int(x0_init.size)
        initial_cost, _grad0, _rnorm0 = objective.eval(x0_init)

    method_key = str(method).strip().lower()
    if method_key not in {"gd", "gn"}:
        raise ValueError(
            "solve_liteopt_gd: liteopt method must be 'gd' or 'gn'. "
            f"Got method={method!r}."
        )

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
        if verbose is not None and "verbose" not in options_local:
            options_local["verbose"] = bool(verbose)
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
            if bool(np.all(np.isfinite(x_star))):
                cost_eval, _grad_eval, rnorm_eval = objective.eval(x_star)
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
        return SolveOutcome(
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

    options_local = {} if options is None else dict(options)
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
    if verbose is not None and "verbose" not in options_local:
        options_local["verbose"] = bool(verbose)
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
        _fx, gx, rnorm = objective.eval(x_vec)
        if not bool(np.isfinite(rnorm)):
            raise ValueError("solve_liteopt_gd: residual norm became non-finite.")
        if not bool(np.all(np.isfinite(gx))):
            raise ValueError("solve_liteopt_gd: gradient became non-finite.")
        last_dxnorm = float(np.linalg.norm(x_vec - prev_x))
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
    with prof.span("solve.finalize"):
        if bool(np.all(np.isfinite(x_star))):
            cost, _grad, rnorm = objective.eval(x_star)
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

    return SolveOutcome(
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


def solve(
    problem: Any,
    *,
    solver: str = "gauss_newton",
    x0: Array | Any = None,
    required: Iterable[StateKey] | None = None,
    on_iter: IterCallback | None = None,
    options: Mapping[str, Any] | None = None,
    profiler: Profiler | None = None,
) -> SolveResult:
    """Dispatch runtime solve to one of: gauss_newton / scipy_minimize / cyipopt / liteopt.

    Returns:
      SolveOutcome(solution, stats, timing, meta)

    Solver parameters are provided via `options`.
    `x0` may be passed directly or as `options["x0"]` (but not both).

    gauss_newton:
      max_iters, tol_r, tol_dx, damping, line_search, ls_beta, ls_min_step, ls_max_iters

    scipy_minimize:
      method, max_iters, tol, bounds, backend_options
      (unknown top-level keys are forwarded to scipy's options dict)

    cyipopt:
      max_iters, tol, bounds, backend_options
      (unknown top-level keys are forwarded to IPOPT options dict)

    liteopt:
      method='gd' (default):
        max_iters, step_size, tol_grad, line_search, verbose,
        nonfinite_retries, nonfinite_step_shrink, min_step_size
      method='gn':
        max_iters, tol_r, tol_dx, lambda_, step_scale,
        damping_update, linear_system,
        line_search_method, line_search, ls_beta, ls_min_step, ls_max_steps,
        verbose
      backend_options
      (unknown top-level keys are forwarded to the selected liteopt backend API)
    """

    key = str(solver).strip().lower()
    if key not in _SOLVER_REI_OPTION_KEYS:
        raise ValueError(
            "Unknown solver. Use one of: "
            "'gauss_newton', 'scipy_minimize', 'cyipopt', 'liteopt'. "
            "Solver aliases are not supported. "
            f"Got solver={solver!r}."
        )

    opts = _merge_options(options)
    if x0 is not None and "x0" in opts:
        raise ValueError("solve: pass x0 either as keyword argument or options['x0'], not both.")
    x0_override = opts.get("x0", x0)
    weighted = bool(opts.get("weighted", True))
    term_indices = _normalize_term_indices_option(opts.get("term_indices", None))
    backend_options = _normalize_backend_options_for_solver(opts, solver_key=key)

    if key == "gauss_newton":
        return solve_gauss_newton(
            problem,
            x0=x0_override,
            required=required,
            weighted=weighted,
            term_indices=term_indices,
            max_iters=int(opts.get("max_iters", 200)),
            tol_r=float(opts.get("tol_r", 1e-10)),
            tol_dx=float(opts.get("tol_dx", 1e-12)),
            damping=float(opts.get("damping", 1e-8)),
            line_search=bool(opts.get("line_search", True)),
            ls_beta=float(opts.get("ls_beta", 0.5)),
            ls_min_step=float(opts.get("ls_min_step", 1e-8)),
            ls_max_iters=int(opts.get("ls_max_iters", 12)),
            on_iter=on_iter,
            profiler=profiler,
        )

    if key == "scipy_minimize":
        tol = opts.get("tol", None)
        return solve_scipy_minimize(
            problem,
            x0=x0_override,
            required=required,
            weighted=weighted,
            term_indices=term_indices,
            method=str(opts.get("method", "L-BFGS-B")),
            max_iters=int(opts.get("max_iters", 200)),
            tol=(None if tol is None else float(tol)),
            bounds=opts.get("bounds", None),
            options=backend_options,
            on_iter=on_iter,
            profiler=profiler,
        )

    if key == "cyipopt":
        tol = opts.get("tol", None)
        return solve_cyipopt_minimize(
            problem,
            x0=x0_override,
            required=required,
            weighted=weighted,
            term_indices=term_indices,
            max_iters=int(opts.get("max_iters", 200)),
            tol=(None if tol is None else float(tol)),
            bounds=opts.get("bounds", None),
            options=backend_options,
            on_iter=on_iter,
            profiler=profiler,
        )

    if key == "liteopt":
        method_liteopt = str(opts.get("method", "gd")).strip().lower()
        verbose_opt = opts.get("verbose", None)
        line_search_opt = opts.get("line_search", None)
        ls_max_steps_raw = opts.get("ls_max_steps", opts.get("ls_max_iters", 12))

        if method_liteopt == "gn":
            return solve_liteopt_gd(
                problem,
                x0=x0_override,
                required=required,
                weighted=weighted,
                term_indices=term_indices,
                method="gn",
                verbose=(
                    None if verbose_opt is None else bool(verbose_opt)
                ),
                max_iters=int(opts.get("max_iters", 200)),
                tol_r=float(opts.get("tol_r", 1e-10)),
                tol_dx=float(opts.get("tol_dx", 1e-12)),
                lambda_=float(opts.get("lambda_", 1e-8)),
                step_scale=(
                    None
                    if opts.get("step_scale", None) is None
                    else float(opts.get("step_scale"))
                ),
                damping_update=str(opts.get("damping_update", "fixed")),
                linear_system=str(opts.get("linear_system", "normal_jtj")),
                line_search_method=str(opts.get("line_search_method", "strict_decrease")),
                line_search=(
                    True if line_search_opt is None else bool(line_search_opt)
                ),
                ls_beta=float(opts.get("ls_beta", 0.5)),
                ls_min_step=float(opts.get("ls_min_step", 1e-8)),
                ls_max_steps=int(ls_max_steps_raw),
                c_armijo=(
                    None
                    if opts.get("c_armijo", None) is None
                    else float(opts.get("c_armijo"))
                ),
                options=backend_options,
                on_iter=on_iter,
                profiler=profiler,
            )

        return solve_liteopt_gd(
            problem,
            x0=x0_override,
            required=required,
            weighted=weighted,
            term_indices=term_indices,
            method=method_liteopt,
            verbose=(
                None if verbose_opt is None else bool(verbose_opt)
            ),
            max_iters=int(opts.get("max_iters", 200)),
            step_size=float(opts.get("step_size", 1e-3)),
            tol_grad=float(opts.get("tol_grad", 1e-4)),
            line_search=(
                None if line_search_opt is None else bool(line_search_opt)
            ),
            nonfinite_retries=int(opts.get("nonfinite_retries", 8)),
            nonfinite_step_shrink=float(opts.get("nonfinite_step_shrink", 0.2)),
            min_step_size=float(opts.get("min_step_size", 1e-12)),
            options=backend_options,
            on_iter=on_iter,
            profiler=profiler,
        )
    raise RuntimeError(f"internal error: unsupported solver dispatch key {key!r}.")

__all__ = [
    "nls",
    "solve",
    "solve_gauss_newton",
    "solve_scipy_minimize",
    "solve_cyipopt_minimize",
    "solve_liteopt_gd",
]
