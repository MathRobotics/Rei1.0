from __future__ import annotations

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
            "max_iters",
            "step_size",
            "tol_grad",
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


def solve_scipy_minimize(
    problem: Any,
    *,
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
        x0 = np.asarray(linear_problem.get_point(), dtype=float).reshape(-1).copy()
        req = linear_problem.required_list(required)
        objective = _LinearizedObjective(linear_problem, required=req)
        n_total = int(x0.size)
        initial_cost, _grad0, _rnorm0 = objective.eval(x0)

    options_local: dict[str, Any] = {} if options is None else dict(options)
    if max_iters is not None and "maxiter" not in options_local:
        options_local["maxiter"] = int(max_iters)

    iter_count = 0
    last_dxnorm = 0.0
    prev_x = x0.copy()

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
            x0=x0,
            jac=jac,
            method=str(method),
            tol=tol,
            bounds=bounds,
            callback=callback,
            options=options_local,
        )

    with prof.span("solve.finalize"):
        x_star = as_vec(getattr(result, "x", x0), expected_size=n_total, name="result.x")
        cost, _grad, rnorm = objective.eval(x_star)
    iters = int(getattr(result, "nit", iter_count))
    if iters <= 0:
        iters = int(iter_count)
    if iter_count <= 0:
        last_dxnorm = float(np.linalg.norm(x_star - x0))
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
        meta={"solver": "scipy_minimize", "method": str(method)},
    )


def solve_cyipopt_minimize(
    problem: Any,
    *,
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
        x0 = np.asarray(linear_problem.get_point(), dtype=float).reshape(-1).copy()
        req = linear_problem.required_list(required)
        objective = _LinearizedObjective(linear_problem, required=req)
        n_total = int(x0.size)
        initial_cost, _grad0, _rnorm0 = objective.eval(x0)

    options_local: dict[str, Any] = {} if options is None else dict(options)
    if max_iters is not None and "max_iter" not in options_local:
        options_local["max_iter"] = int(max_iters)

    iter_count = 0
    last_dxnorm = 0.0
    prev_x = x0.copy()

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
        "x0": x0,
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
        x_star = as_vec(getattr(result, "x", x0), expected_size=n_total, name="result.x")
        cost, _grad, rnorm = objective.eval(x_star)
    iters = int(getattr(result, "nit", iter_count))
    if iters <= 0:
        iters = int(iter_count)
    if iter_count <= 0:
        last_dxnorm = float(np.linalg.norm(x_star - x0))
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
        meta={"solver": "cyipopt_minimize"},
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


def solve_liteopt_gd(
    problem: Any,
    *,
    required: Iterable[StateKey] | None = None,
    weighted: bool = True,
    term_indices: Iterable[int] | None = None,
    max_iters: int | None = 200,
    step_size: float = 1e-3,
    tol_grad: float = 1e-4,
    options: Mapping[str, Any] | None = None,
    on_iter: IterCallback | None = None,
    profiler: Profiler | None = None,
) -> SolveResult:
    """Solve `||r(x)||^2` via liteopt.gd and return SolveOutcome."""

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
        x0 = np.asarray(linear_problem.get_point(), dtype=float).reshape(-1).copy()
        req = linear_problem.required_list(required)
        objective = _LinearizedObjective(linear_problem, required=req)
        n_total = int(x0.size)
        initial_cost, _grad0, _rnorm0 = objective.eval(x0)

    options_local: dict[str, Any] = {} if options is None else dict(options)
    if "step_size" not in options_local:
        options_local["step_size"] = float(step_size)
    if max_iters is not None and "max_iters" not in options_local:
        options_local["max_iters"] = int(max_iters)
    if "tol_grad" not in options_local:
        options_local["tol_grad"] = float(tol_grad)

    iter_count = 0
    last_dxnorm = 0.0
    prev_x = x0.copy()

    def fun(x: Array) -> float:
        fx, _gx, _rnorm = objective.eval(x)
        return float(fx)

    def grad(x: Array) -> Array:
        nonlocal iter_count, last_dxnorm, prev_x
        x_vec = as_vec(x, expected_size=n_total, name="x")
        _fx, gx, rnorm = objective.eval(x_vec)
        last_dxnorm = float(np.linalg.norm(x_vec - prev_x))
        prev_x = x_vec.copy()
        if on_iter is not None:
            on_iter(iter_count, rnorm, last_dxnorm)
        iter_count += 1
        return gx

    with prof.span("solve.backend"):
        result = liteopt.gd(fun, grad, x0.copy(), **options_local)
    x_star, converged, iters_from_result = _parse_liteopt_gd_result(
        result,
        n_total=n_total,
    )
    with prof.span("solve.finalize"):
        cost, _grad, rnorm = objective.eval(x_star)

    iters = int(iter_count)
    if iters_from_result is not None:
        iters = int(iters_from_result)
    elif iters <= 0:
        iters = int(max_iters) if max_iters is not None else 0

    if iter_count <= 0:
        last_dxnorm = float(np.linalg.norm(x_star - x0))

    status = "converged" if bool(converged) else "failed"
    return SolveOutcome(
        solution=x_star.copy(),
        stats=SolveStats(
            status=status,
            iterations=iters,
            initial_objective=float(initial_cost),
            objective=float(cost),
            residual_norm=float(rnorm),
            step_norm=float(last_dxnorm),
        ),
        timing=prof.snapshot(),
        meta={"solver": "liteopt_gd"},
    )


def solve(
    problem: Any,
    *,
    solver: str = "gauss_newton",
    required: Iterable[StateKey] | None = None,
    on_iter: IterCallback | None = None,
    options: Mapping[str, Any] | None = None,
    profiler: Profiler | None = None,
) -> SolveResult:
    """Dispatch runtime solve to one of: gauss_newton / scipy_minimize / cyipopt / liteopt.

    Returns:
      SolveOutcome(solution, stats, timing, meta)

    Solver parameters are provided via `options`.

    gauss_newton:
      max_iters, tol_r, tol_dx, damping, line_search, ls_beta, ls_min_step, ls_max_iters

    scipy_minimize:
      method, max_iters, tol, bounds, backend_options
      (unknown top-level keys are forwarded to scipy's options dict)

    cyipopt:
      max_iters, tol, bounds, backend_options
      (unknown top-level keys are forwarded to IPOPT options dict)

    liteopt:
      max_iters, step_size, tol_grad, backend_options
      (unknown top-level keys are forwarded to liteopt.gd kwargs)
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
    weighted = bool(opts.get("weighted", True))
    term_indices = _normalize_term_indices_option(opts.get("term_indices", None))
    backend_options = _normalize_backend_options_for_solver(opts, solver_key=key)

    if key == "gauss_newton":
        return solve_gauss_newton(
            problem,
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
        return solve_liteopt_gd(
            problem,
            required=required,
            weighted=weighted,
            term_indices=term_indices,
            max_iters=int(opts.get("max_iters", 200)),
            step_size=float(opts.get("step_size", 1e-3)),
            tol_grad=float(opts.get("tol_grad", 1e-4)),
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
