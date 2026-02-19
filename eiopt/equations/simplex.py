from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.state_cache import StateKey
from ..problem import LinearizedProblem, ProjectProblem
from ..xops import as_vec

Array = np.ndarray


def _project_to_simplex(v: Array) -> Array:
    x = np.asarray(v, dtype=float).reshape(-1)
    n = int(x.size)
    if n == 0:
        return np.zeros((0,), dtype=float)

    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1, dtype=float)
    cond = u - cssv / ind > 0.0
    if not np.any(cond):
        return np.full((n,), 1.0 / n, dtype=float)
    rho = int(np.nonzero(cond)[0][-1])
    theta = float(cssv[rho] / (rho + 1))
    return np.maximum(x - theta, 0.0)


def _default_step_size_from_jacobian(J: Array) -> float:
    J_mat = np.asarray(J, dtype=float)
    svals = np.linalg.svd(J_mat, compute_uv=False)
    lipschitz = float((svals[0] ** 2) if svals.size > 0 else 1.0)
    lipschitz = max(lipschitz, 1e-12)
    return float(1.0 / lipschitz)


def _linearize_checked(
    problem: LinearizedProblem,
    *,
    required: Iterable[StateKey] | None,
    n_total: int,
) -> tuple[Array, Array]:
    r_raw, J_raw = problem.linearize(required=required)
    r = np.asarray(r_raw, dtype=float).reshape(-1)
    J = np.asarray(J_raw, dtype=float)
    if J.ndim != 2:
        raise ValueError(
            f"solve_projected_linearized_min_norm: jacobian must be 2D, got shape {J.shape}."
        )
    if J.shape[0] != r.size:
        raise ValueError(
            "solve_projected_linearized_min_norm: row mismatch between residual and jacobian. "
            f"len(r)={r.size}, J.shape={J.shape}."
        )
    if J.shape[1] != int(n_total):
        raise ValueError(
            "solve_projected_linearized_min_norm: jacobian column mismatch. "
            f"expected {int(n_total)}, got {J.shape[1]}."
        )
    return r, J


@dataclass
class SimplexMinNormProblem:
    """Linearized+project problem for `min ||A @ x||^2` on the simplex."""

    A: Array | Any
    x: Array | Any = None

    def __post_init__(self) -> None:
        A_mat = np.asarray(self.A, dtype=float)
        if A_mat.ndim != 2:
            raise ValueError(f"SimplexMinNormProblem: A must be 2D, got shape {A_mat.shape}.")
        _m, n = A_mat.shape
        n_i = int(n)
        if n_i <= 0:
            raise ValueError("SimplexMinNormProblem: A must have at least one column.")
        self.A = A_mat

        if self.x is None:
            self.x = np.full((n_i,), 1.0 / n_i, dtype=float)
        else:
            x_init = np.asarray(self.x, dtype=float).reshape(-1)
            if x_init.size != n_i:
                raise ValueError(
                    f"SimplexMinNormProblem: x size mismatch. Expected {n_i}, got {x_init.size}."
                )
            self.x = _project_to_simplex(x_init)

    @property
    def n_total(self) -> int:
        return int(np.asarray(self.A, dtype=float).shape[1])

    def get_point(self) -> Array:
        return np.asarray(self.x, dtype=float).reshape(-1).copy()

    def set_point(self, x: Array | Any) -> None:
        self.x = as_vec(x, expected_size=int(self.n_total), name="x").copy()

    def required_list(self, required: Iterable[StateKey] | None = None) -> list[StateKey]:
        if required is None:
            return []
        return list(required)

    def eval(self, *, required: Iterable[StateKey] | None = None) -> Array:
        del required
        x = np.asarray(self.x, dtype=float).reshape(-1)
        return np.asarray(self.A, dtype=float) @ x

    def linearize(self, *, required: Iterable[StateKey] | None = None) -> tuple[Array, Array]:
        del required
        r = self.eval()
        return np.asarray(r, dtype=float).reshape(-1), np.asarray(self.A, dtype=float)

    def project(self, x: Array | Any) -> Array:
        x_vec = as_vec(x, expected_size=int(self.n_total), name="x")
        return _project_to_simplex(x_vec)


def solve_projected_linearized_min_norm(
    problem: LinearizedProblem,
    projector: ProjectProblem,
    *,
    max_iters: int = 2000,
    tol: float = 1e-10,
    step_size: float | None = None,
    x0: Array | Any = None,
    required: Iterable[StateKey] | None = None,
    return_info: bool = False,
) -> Array | tuple[Array, dict[str, Any]]:
    """Solve `min ||r(x)||^2` using projected gradient with linearized residuals."""

    n_total = int(problem.n_total)
    if int(projector.n_total) != n_total:
        raise ValueError(
            "solve_projected_linearized_min_norm: problem/projector size mismatch. "
            f"problem.n_total={n_total}, projector.n_total={int(projector.n_total)}."
        )

    max_iters_i = int(max_iters)
    if max_iters_i <= 0:
        raise ValueError(
            f"solve_projected_linearized_min_norm: max_iters must be > 0, got {max_iters_i}."
        )
    tol_f = float(tol)
    if tol_f <= 0.0:
        raise ValueError(f"solve_projected_linearized_min_norm: tol must be > 0, got {tol_f}.")

    req = problem.required_list(required)
    x_init = problem.get_point() if x0 is None else as_vec(x0, expected_size=n_total, name="x0")
    x = as_vec(projector.project(x_init), expected_size=n_total, name="project(x0)")
    problem.set_point(x)
    _r0, J0 = _linearize_checked(problem, required=req, n_total=n_total)

    if step_size is None:
        step = _default_step_size_from_jacobian(J0)
    else:
        step = float(step_size)
        if step <= 0.0:
            raise ValueError(
                f"solve_projected_linearized_min_norm: step_size must be > 0, got {step}."
            )

    converged = False
    iterations = max_iters_i
    for it in range(max_iters_i):
        problem.set_point(x)
        r, J = _linearize_checked(problem, required=req, n_total=n_total)
        grad = np.asarray(J.T @ r, dtype=float).reshape(-1)
        x_next = as_vec(projector.project(x - step * grad), expected_size=n_total, name="project(x)")
        if float(np.linalg.norm(x_next - x)) <= tol_f:
            x = x_next
            converged = True
            iterations = it + 1
            break
        x = x_next

    problem.set_point(x)
    r_fin, J_fin = _linearize_checked(problem, required=req, n_total=n_total)
    grad_fin = np.asarray(J_fin.T @ r_fin, dtype=float).reshape(-1)
    objective = float(0.5 * (r_fin @ r_fin))

    if not return_info:
        return x

    info = {
        "converged": converged,
        "iterations": int(iterations),
        "objective": objective,
        "residual_norm": float(np.linalg.norm(r_fin)),
        "grad_norm": float(np.linalg.norm(grad_fin)),
        "step_size": float(step),
    }
    return x, info


def solve_simplex_min_norm(
    A: Array | Any,
    *,
    max_iters: int = 2000,
    tol: float = 1e-10,
    step_size: float | None = None,
    x0: Array | Any = None,
    return_info: bool = False,
) -> Array | tuple[Array, dict[str, Any]]:
    """Solve simplex-constrained coefficients by `min ||A @ x||^2`."""

    A_mat = np.asarray(A, dtype=float)
    if A_mat.ndim != 2:
        raise ValueError(f"solve_simplex_min_norm: A must be 2D, got shape {A_mat.shape}.")
    _m, n = A_mat.shape
    n_i = int(n)
    if n_i <= 0:
        raise ValueError("solve_simplex_min_norm: A must have at least one column.")

    max_iters_i = int(max_iters)
    if max_iters_i <= 0:
        raise ValueError(f"solve_simplex_min_norm: max_iters must be > 0, got {max_iters_i}.")
    tol_f = float(tol)
    if tol_f <= 0.0:
        raise ValueError(f"solve_simplex_min_norm: tol must be > 0, got {tol_f}.")

    if n_i == 1:
        w_single = np.array([1.0], dtype=float)
        if not return_info:
            return w_single
        r = A_mat @ w_single
        info = {
            "converged": True,
            "iterations": 0,
            "objective": float(0.5 * (r @ r)),
            "residual_norm": float(np.linalg.norm(r)),
            "grad_norm": float(np.linalg.norm(A_mat.T @ r)),
            "step_size": 0.0,
        }
        return w_single, info

    x_init: Array | Any
    if x0 is None:
        x_init = np.full((n_i,), 1.0 / n_i, dtype=float)
    else:
        x_arr = np.asarray(x0, dtype=float).reshape(-1)
        if x_arr.size != n_i:
            raise ValueError(
                f"solve_simplex_min_norm: x0 size mismatch. Expected {n_i}, got {x_arr.size}."
            )
        x_init = _project_to_simplex(x_arr)

    problem = SimplexMinNormProblem(A=A_mat, x=x_init)
    return solve_projected_linearized_min_norm(
        problem,
        problem,
        max_iters=max_iters_i,
        tol=tol_f,
        step_size=step_size,
        return_info=return_info,
    )


__all__ = [
    "SimplexMinNormProblem",
    "solve_projected_linearized_min_norm",
    "solve_simplex_min_norm",
]
