from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from ...core.state_cache import StateKey
from .._xops import as_vec, set_runtime_x
from ..runtime import NLSRuntime
from .gauss_newton import solve_gauss_newton
from .nls import nls

Array = np.ndarray
SolveResult = tuple[Array, float, float, int, float, float, bool]
IterCallback = Callable[[int, float, float], None]


class _RuntimeObjective:
    def __init__(
        self,
        runtime: NLSRuntime,
        *,
        required: Iterable[StateKey] | None = None,
    ) -> None:
        self.runtime = runtime
        self.required = None if required is None else list(required)
        self._last_x: Array | None = None
        self._last_cost: float = float("nan")
        self._last_grad: Array | None = None
        self._last_rnorm: float = float("inf")

    def eval(self, x: Array) -> tuple[float, Array, float]:
        x_vec = as_vec(x, expected_size=int(self.runtime.pack.n_total), name="x")
        if self._last_x is not None and np.array_equal(self._last_x, x_vec):
            if self._last_grad is None:
                raise RuntimeError("internal error: gradient cache is missing.")
            return self._last_cost, self._last_grad, self._last_rnorm

        set_runtime_x(self.runtime, x_vec, name="x")
        r_all, J_all = self.runtime.linearize(required=self.required)
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


def solve_scipy_minimize(
    runtime: NLSRuntime,
    *,
    required: Iterable[StateKey] | None = None,
    method: str = "L-BFGS-B",
    max_iters: int | None = 200,
    tol: float | None = None,
    bounds: Any = None,
    options: Mapping[str, Any] | None = None,
    on_iter: IterCallback | None = None,
) -> SolveResult:
    """Solve `||r(x)||^2` from a `NLSRuntime` via scipy.optimize.minimize."""

    try:
        from scipy.optimize import minimize
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "solve_scipy_minimize requires scipy. Install scipy and re-run."
        ) from e

    x0 = np.asarray(runtime.pack.get(), dtype=float).reshape(-1).copy()
    req = runtime.required_list(required)
    objective = _RuntimeObjective(runtime, required=req)
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

    x_star = as_vec(getattr(result, "x", x0), expected_size=n_total, name="result.x")
    cost, _grad, rnorm = objective.eval(x_star)
    iters = int(getattr(result, "nit", iter_count))
    if iters <= 0:
        iters = int(iter_count)
    if iter_count <= 0:
        last_dxnorm = float(np.linalg.norm(x_star - x0))
    converged = bool(getattr(result, "success", False))

    return x_star.copy(), float(initial_cost), float(cost), iters, float(rnorm), float(last_dxnorm), converged


def solve_cyipopt_minimize(
    runtime: NLSRuntime,
    *,
    required: Iterable[StateKey] | None = None,
    max_iters: int | None = 200,
    tol: float | None = None,
    bounds: Any = None,
    options: Mapping[str, Any] | None = None,
    on_iter: IterCallback | None = None,
) -> SolveResult:
    """Solve `||r(x)||^2` from a `NLSRuntime` via cyipopt.minimize_ipopt."""

    try:
        from cyipopt import minimize_ipopt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "solve_cyipopt_minimize requires cyipopt. Install cyipopt and re-run."
        ) from e

    x0 = np.asarray(runtime.pack.get(), dtype=float).reshape(-1).copy()
    req = runtime.required_list(required)
    objective = _RuntimeObjective(runtime, required=req)
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

    try:
        result = minimize_ipopt(**kwargs)
    except TypeError as e:
        if "callback" not in str(e):
            raise
        kwargs.pop("callback", None)
        result = minimize_ipopt(**kwargs)

    x_star = as_vec(getattr(result, "x", x0), expected_size=n_total, name="result.x")
    cost, _grad, rnorm = objective.eval(x_star)
    iters = int(getattr(result, "nit", iter_count))
    if iters <= 0:
        iters = int(iter_count)
    if iter_count <= 0:
        last_dxnorm = float(np.linalg.norm(x_star - x0))
    converged = bool(getattr(result, "success", False))

    return x_star.copy(), float(initial_cost), float(cost), iters, float(rnorm), float(last_dxnorm), converged


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
    runtime: NLSRuntime,
    *,
    required: Iterable[StateKey] | None = None,
    max_iters: int | None = 200,
    step_size: float = 1e-3,
    tol_grad: float = 1e-4,
    options: Mapping[str, Any] | None = None,
    on_iter: IterCallback | None = None,
) -> SolveResult:
    """Solve `||r(x)||^2` from a `NLSRuntime` via liteopt.gd."""

    try:
        import liteopt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "solve_liteopt_gd requires liteopt. Install liteopt and re-run."
        ) from e

    x0 = np.asarray(runtime.pack.get(), dtype=float).reshape(-1).copy()
    req = runtime.required_list(required)
    objective = _RuntimeObjective(runtime, required=req)
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

    result = liteopt.gd(fun, grad, x0.copy(), **options_local)
    x_star, converged, iters_from_result = _parse_liteopt_gd_result(
        result,
        n_total=n_total,
    )
    cost, _grad, rnorm = objective.eval(x_star)

    iters = int(iter_count)
    if iters_from_result is not None:
        iters = int(iters_from_result)
    elif iters <= 0:
        iters = int(max_iters) if max_iters is not None else 0

    if iter_count <= 0:
        last_dxnorm = float(np.linalg.norm(x_star - x0))

    return (
        x_star.copy(),
        float(initial_cost),
        float(cost),
        iters,
        float(rnorm),
        float(last_dxnorm),
        bool(converged),
    )


def solve(
    runtime: NLSRuntime,
    *,
    solver: str = "gauss_newton",
    required: Iterable[StateKey] | None = None,
    on_iter: IterCallback | None = None,
    options: Mapping[str, Any] | None = None,
) -> SolveResult:
    """Dispatch runtime solve to one of: gauss_newton / scipy / cyipopt / liteopt.

    Solver parameters are provided via `options`.

    gauss_newton:
      max_iters, tol_r, tol_dx, damping, line_search, ls_beta, ls_min_step, ls_max_iters

    scipy:
      method, max_iters, tol, bounds, backend_options

    cyipopt:
      max_iters, tol, bounds, backend_options

    liteopt:
      max_iters, step_size, tol_grad, backend_options
    """

    key = str(solver).strip().lower()
    opts = _merge_options(options)

    if key in {"gauss_newton", "gauss-newton", "gn"}:
        return solve_gauss_newton(
            runtime,
            required=required,
            max_iters=int(opts.get("max_iters", 200)),
            tol_r=float(opts.get("tol_r", 1e-10)),
            tol_dx=float(opts.get("tol_dx", 1e-12)),
            damping=float(opts.get("damping", 1e-8)),
            line_search=bool(opts.get("line_search", True)),
            ls_beta=float(opts.get("ls_beta", 0.5)),
            ls_min_step=float(opts.get("ls_min_step", 1e-8)),
            ls_max_iters=int(opts.get("ls_max_iters", 12)),
            on_iter=on_iter,
        )

    if key in {"scipy", "scipy_minimize", "minimize", "scipy.optimize.minimize"}:
        tol = opts.get("tol", None)
        return solve_scipy_minimize(
            runtime,
            required=required,
            method=str(opts.get("method", "L-BFGS-B")),
            max_iters=int(opts.get("max_iters", 200)),
            tol=(None if tol is None else float(tol)),
            bounds=opts.get("bounds", None),
            options=_as_options_mapping(opts.get("backend_options", None), where="solve(scipy)"),
            on_iter=on_iter,
        )

    if key in {"cyipopt", "ipopt", "minimize_ipopt", "cyipopt.minimize_ipopt"}:
        tol = opts.get("tol", None)
        return solve_cyipopt_minimize(
            runtime,
            required=required,
            max_iters=int(opts.get("max_iters", 200)),
            tol=(None if tol is None else float(tol)),
            bounds=opts.get("bounds", None),
            options=_as_options_mapping(opts.get("backend_options", None), where="solve(cyipopt)"),
            on_iter=on_iter,
        )

    if key in {"liteopt", "liteopt_gd", "gd", "liteopt.gd"}:
        return solve_liteopt_gd(
            runtime,
            required=required,
            max_iters=int(opts.get("max_iters", 200)),
            step_size=float(opts.get("step_size", 1e-3)),
            tol_grad=float(opts.get("tol_grad", 1e-4)),
            options=_as_options_mapping(opts.get("backend_options", None), where="solve(liteopt)"),
            on_iter=on_iter,
        )

    raise ValueError(
        "Unknown solver. Use one of: "
        "'gauss_newton', 'scipy_minimize', 'cyipopt', 'liteopt'. "
        f"Got solver={solver!r}."
    )

__all__ = [
    "nls",
    "solve",
    "solve_gauss_newton",
    "solve_scipy_minimize",
    "solve_cyipopt_minimize",
    "solve_liteopt_gd",
]
