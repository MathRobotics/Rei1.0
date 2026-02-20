from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.outcome import SolveOutcome, SolveStats
from ..core.state_cache import StateKey
from ..core.timing import Profiler, ensure_profiler
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
    profiler: Profiler | None = None,
) -> SolveOutcome:
    """Solve `min ||r(x)||^2` using projected gradient with linearized residuals."""

    prof = ensure_profiler(profiler)
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

    with prof.span("solve.setup"):
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
    step_norm_last = float("inf")
    for it in range(max_iters_i):
        with prof.span("solve.iter.linearize"):
            problem.set_point(x)
            r, J = _linearize_checked(problem, required=req, n_total=n_total)
        with prof.span("solve.iter.step"):
            grad = np.asarray(J.T @ r, dtype=float).reshape(-1)
            x_next = as_vec(projector.project(x - step * grad), expected_size=n_total, name="project(x)")
            step_norm_last = float(np.linalg.norm(x_next - x))
        if step_norm_last <= tol_f:
            x = x_next
            converged = True
            iterations = it + 1
            break
        x = x_next

    with prof.span("solve.finalize"):
        problem.set_point(x)
        r_fin, J_fin = _linearize_checked(problem, required=req, n_total=n_total)
        grad_fin = np.asarray(J_fin.T @ r_fin, dtype=float).reshape(-1)
        objective = float(0.5 * (r_fin @ r_fin))
        residual_norm = float(np.linalg.norm(r_fin))
        grad_norm = float(np.linalg.norm(grad_fin))

    return SolveOutcome(
        solution=x,
        stats=SolveStats(
            status=("converged" if converged else "max_iters"),
            iterations=int(iterations),
            objective=float(objective),
            residual_norm=float(residual_norm),
            step_norm=float(step_norm_last),
        ),
        timing=prof.snapshot(),
        meta={
            "solver": "projected_linearized_min_norm",
            "step_size": float(step),
            "grad_norm": float(grad_norm),
        },
    )


def _normalize_simplex_method_name(method: str) -> str:
    key = str(method).strip().lower().replace("-", "_")
    if key == "projected_gradient":
        return "projected_gradient"
    if key == "qr_nullspace":
        return "qr_nullspace"
    raise ValueError(
        "solve_simplex_min_norm: method must be one of "
        "'projected_gradient' or 'qr_nullspace'. "
        "Method aliases are not supported. "
        f"Got method={method!r}."
    )


def _solve_simplex_min_norm_qr_nullspace(
    A_mat: Array,
    *,
    max_iters: int,
    tol: float,
) -> tuple[Array, dict[str, Any]]:
    """Solve simplex min-norm via active-set + QR nullspace elimination."""

    _m, n = A_mat.shape
    n_i = int(n)
    if n_i <= 0:
        raise ValueError("_solve_simplex_min_norm_qr_nullspace: A must have at least one column.")

    if n_i == 1:
        w_single = np.array([1.0], dtype=float)
        r_single = A_mat @ w_single
        return w_single, {
            "converged": True,
            "iterations": 0,
            "objective": float(0.5 * (r_single @ r_single)),
            "residual_norm": float(np.linalg.norm(r_single)),
            "grad_norm": float(np.linalg.norm(A_mat.T @ r_single)),
            "step_size": 0.0,
            "method": "qr_nullspace",
            "active_size": 1,
        }

    max_iters_i = int(max_iters)
    if max_iters_i <= 0:
        raise ValueError(
            "_solve_simplex_min_norm_qr_nullspace: max_iters must be > 0, "
            f"got {max_iters_i}."
        )
    tol_f = float(tol)
    if tol_f <= 0.0:
        raise ValueError(
            "_solve_simplex_min_norm_qr_nullspace: tol must be > 0, "
            f"got {tol_f}."
        )

    active = np.arange(n_i, dtype=int)
    w = np.full((n_i,), 1.0 / float(n_i), dtype=float)
    converged = False
    iterations = 0

    for it in range(min(max_iters_i, n_i + 1)):
        iterations = it + 1
        k = int(active.size)
        if k <= 0:
            break
        if k == 1:
            w = np.zeros((n_i,), dtype=float)
            w[int(active[0])] = 1.0
            converged = True
            break

        A_act = np.asarray(A_mat[:, active], dtype=float)
        ones = np.ones((k,), dtype=float)
        # w = w0 + N z, where N spans null(ones^T) and w0 satisfies ones^T w0 = 1.
        Q, _R = np.linalg.qr(ones.reshape(-1, 1), mode="complete")
        N = np.asarray(Q[:, 1:], dtype=float)
        w0 = ones / float(k)

        B = A_act @ N
        b = A_act @ w0
        if B.shape[1] == 0:
            z = np.zeros((0,), dtype=float)
        else:
            z, *_ = np.linalg.lstsq(B, -b, rcond=None)
            z = np.asarray(z, dtype=float).reshape(-1)
        w_act = np.asarray(w0 + N @ z, dtype=float).reshape(-1)

        i_min = int(np.argmin(w_act))
        if float(w_act[i_min]) >= -tol_f:
            w_act = np.maximum(w_act, 0.0)
            s = float(w_act.sum())
            if s <= 0.0:
                w_act = ones / float(k)
            else:
                w_act = w_act / s
            w = np.zeros((n_i,), dtype=float)
            w[active] = w_act
            converged = True
            break

        drop = int(active[i_min])
        active = active[active != drop]

    w = _project_to_simplex(w)
    r = A_mat @ w
    info = {
        "converged": bool(converged),
        "iterations": int(iterations),
        "objective": float(0.5 * (r @ r)),
        "residual_norm": float(np.linalg.norm(r)),
        "grad_norm": float(np.linalg.norm(A_mat.T @ r)),
        "step_size": 0.0,
        "method": "qr_nullspace",
        "active_size": int(active.size),
    }
    return w, info


def solve_simplex_min_norm(
    A: Array | Any,
    *,
    max_iters: int = 2000,
    tol: float = 1e-10,
    step_size: float | None = None,
    x0: Array | Any = None,
    method: str = "projected_gradient",
    profiler: Profiler | None = None,
) -> SolveOutcome:
    """Solve simplex-constrained coefficients by `min ||A @ x||^2`."""

    prof = ensure_profiler(profiler)
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
    method_name = _normalize_simplex_method_name(method)

    if n_i == 1:
        w_single = np.array([1.0], dtype=float)
        with prof.span("solve.finalize"):
            r = A_mat @ w_single
            objective = float(0.5 * (r @ r))
            residual_norm = float(np.linalg.norm(r))
            grad_norm = float(np.linalg.norm(A_mat.T @ r))
        return SolveOutcome(
            solution=w_single,
            stats=SolveStats(
                status="converged",
                iterations=0,
                objective=objective,
                residual_norm=residual_norm,
                step_norm=0.0,
            ),
            timing=prof.snapshot(),
            meta={
                "solver": "simplex_min_norm",
                "method": method_name,
                "grad_norm": grad_norm,
                "step_size": 0.0,
            },
        )

    if method_name == "qr_nullspace":
        with prof.span("solve.backend"):
            w_qr, info_qr = _solve_simplex_min_norm_qr_nullspace(
                A_mat,
                max_iters=max_iters_i,
                tol=tol_f,
            )
        return SolveOutcome(
            solution=w_qr,
            stats=SolveStats(
                status=("converged" if bool(info_qr.get("converged", False)) else "max_iters"),
                iterations=int(info_qr.get("iterations", 0)),
                objective=float(info_qr.get("objective", float("nan"))),
                residual_norm=float(info_qr.get("residual_norm", float("nan"))),
                step_norm=0.0,
            ),
            timing=prof.snapshot(),
            meta={
                "solver": "simplex_min_norm",
                "method": "qr_nullspace",
                "grad_norm": float(info_qr.get("grad_norm", float("nan"))),
                "step_size": 0.0,
                "active_size": int(info_qr.get("active_size", 0)),
            },
        )

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
    out = solve_projected_linearized_min_norm(
        problem,
        problem,
        max_iters=max_iters_i,
        tol=tol_f,
        step_size=step_size,
        profiler=prof,
    )
    meta_local = dict(out.meta)
    meta_local["solver"] = "simplex_min_norm"
    meta_local["method"] = "projected_gradient"
    return SolveOutcome(
        solution=np.asarray(out.solution, dtype=float).reshape(-1).copy(),
        stats=out.stats,
        timing=out.timing,
        meta=meta_local,
    )


__all__ = [
    "SimplexMinNormProblem",
    "solve_projected_linearized_min_norm",
    "solve_simplex_min_norm",
]
