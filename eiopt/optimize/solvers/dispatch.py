from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from ...core.state_cache import StateKey
from ..runtime import NLSRuntime
from .gauss_newton import solve_gauss_newton
from .nls import nls

Array = np.ndarray
SolveResult = tuple[Array, float, float, int, float, float, bool]
IterCallback = Callable[[int, float, float], None]


def _as_vec(x: Array | Any, *, expected_size: int, name: str) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size != int(expected_size):
        raise ValueError(f"{name}: expected size={int(expected_size)}, got size={v.size}.")
    return v


def _set_runtime_x(runtime: NLSRuntime, x: Array) -> None:
    pack = runtime.pack
    x_new = _as_vec(x, expected_size=int(pack.n_total), name="x")
    x_cur = np.asarray(pack.get(), dtype=float).reshape(-1)
    if np.array_equal(x_cur, x_new):
        return
    pack.apply_dx(x_new - x_cur)


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
        x_vec = _as_vec(x, expected_size=int(self.runtime.pack.n_total), name="x")
        if self._last_x is not None and np.array_equal(self._last_x, x_vec):
            if self._last_grad is None:
                raise RuntimeError("internal error: gradient cache is missing.")
            return self._last_cost, self._last_grad, self._last_rnorm

        _set_runtime_x(self.runtime, x_vec)
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
        return _as_vec(xk, expected_size=size, name="callback.x")
    except Exception:
        pass

    x_attr = getattr(xk, "x", None)
    if x_attr is None:
        return None
    try:
        return _as_vec(x_attr, expected_size=size, name="callback.x")
    except Exception:
        return None


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

    x_star = _as_vec(getattr(result, "x", x0), expected_size=n_total, name="result.x")
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

    x_star = _as_vec(getattr(result, "x", x0), expected_size=n_total, name="result.x")
    cost, _grad, rnorm = objective.eval(x_star)
    iters = int(getattr(result, "nit", iter_count))
    if iters <= 0:
        iters = int(iter_count)
    if iter_count <= 0:
        last_dxnorm = float(np.linalg.norm(x_star - x0))
    converged = bool(getattr(result, "success", False))

    return x_star.copy(), float(initial_cost), float(cost), iters, float(rnorm), float(last_dxnorm), converged


def solve(
    runtime: NLSRuntime,
    *,
    solver: str = "gauss_newton",
    required: Iterable[StateKey] | None = None,
    max_iters: int = 200,
    tol_r: float = 1e-10,
    tol_dx: float = 1e-12,
    gn_damping: float = 1e-8,
    gn_line_search: bool = True,
    gn_ls_beta: float = 0.5,
    gn_ls_min_step: float = 1e-8,
    gn_ls_max_iters: int = 12,
    tol: float | None = None,
    on_iter: IterCallback | None = None,
    scipy_method: str = "L-BFGS-B",
    scipy_options: Mapping[str, Any] | None = None,
    scipy_bounds: Any = None,
    ipopt_options: Mapping[str, Any] | None = None,
    ipopt_bounds: Any = None,
) -> SolveResult:
    """Dispatch runtime solve to one of: gauss_newton / scipy / cyipopt."""

    key = str(solver).strip().lower()
    if key in {"gauss_newton", "gauss-newton", "gn"}:
        return solve_gauss_newton(
            runtime,
            max_iters=int(max_iters),
            required=required,
            tol_r=float(tol_r),
            tol_dx=float(tol_dx),
            damping=float(gn_damping),
            line_search=bool(gn_line_search),
            ls_beta=float(gn_ls_beta),
            ls_min_step=float(gn_ls_min_step),
            ls_max_iters=int(gn_ls_max_iters),
            on_iter=on_iter,
        )

    if key in {"scipy", "scipy_minimize", "minimize", "scipy.optimize.minimize"}:
        return solve_scipy_minimize(
            runtime,
            required=required,
            method=scipy_method,
            max_iters=int(max_iters),
            tol=tol,
            bounds=scipy_bounds,
            options=scipy_options,
            on_iter=on_iter,
        )

    if key in {"cyipopt", "ipopt", "minimize_ipopt", "cyipopt.minimize_ipopt"}:
        return solve_cyipopt_minimize(
            runtime,
            required=required,
            max_iters=int(max_iters),
            tol=tol,
            bounds=ipopt_bounds,
            options=ipopt_options,
            on_iter=on_iter,
        )

    raise ValueError(
        "Unknown solver. Use one of: "
        "'gauss_newton', 'scipy_minimize', 'cyipopt'. "
        f"Got solver={solver!r}."
    )

__all__ = [
    "nls",
    "solve",
    "solve_gauss_newton",
    "solve_scipy_minimize",
    "solve_cyipopt_minimize",
]
