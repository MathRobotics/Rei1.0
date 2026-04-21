from __future__ import annotations

from pathlib import Path

import numpy as np

from rei import (
    RuntimeStationaritySource,
    build_stationarity_gradient_matrix,
    format_ioc_report,
    filter_stationarity_contributions,
    select_active_stationarity_indices,
    solve_simplex_min_norm,
)
from rei.optimize.builder import load_problem_toml
from rei.optimize.kkt import check_kkt_conditions
from rei.optimize.reductions import build_nullspace_equality_reduction
from rei.optimize.solvers import solve
from rei.optimize_backends.kots import compile_kots_trajectory_problem

try:
    from robokots.kots import Kots
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires RoboKots.\n"
        "Install dependencies (e.g. `uv sync --group kots`) and re-run:\n"
        "  PYTHONPATH=. python examples/11_forward_then_inverse_ioc.py"
    ) from e

_EXAMPLES_DIR = Path(__file__).resolve().parent
# _MODEL_PATH = _EXAMPLES_DIR / "models" / "planar2.json"
# _MODEL_PATH = _EXAMPLES_DIR / "models" / "sample_robot.json"
# _MODEL_PATH = _EXAMPLES_DIR / "models" / "sample_robot.urdf"
# _MODEL_PATH = _EXAMPLES_DIR / "models" / "7_dof_arm.json"
_MODEL_PATH = _EXAMPLES_DIR / "models" / "7_dof_arm.urdf"
_DSL_PATH = _EXAMPLES_DIR / "dsl" / "robokots_traj_dynamics_d12.toml"
# _DSL_PATH = _EXAMPLES_DIR / "dsl" / "robokots_traj_dynamics.toml"  # up to torque_d1
_ORDER = 4
_NULLSPACE_EQ_SELECTOR_ATTR = "nullspace_eq"

# Simple fixed settings (edit directly if needed)
_FORWARD_SOLVER = "gauss_newton"  # "gauss_newton" | "scipy_minimize" | "cyipopt" | "liteopt"
if _FORWARD_SOLVER == "gauss_newton":
    _FORWARD_OPTIONS = {
        "max_iters": 500,
        "tol_dx": 1e-12,
        "line_search": True,
        "ls_beta": 0.5,
        "ls_min_step": 1e-10,
        "ls_max_iters": 50,
    }
elif _FORWARD_SOLVER == "liteopt":
    _FORWARD_OPTIONS = {
        "method": "gn",
        "max_iters": 300,
        "tol_r": 1e-8,
        "tol_dx": 1e-10,
        "lambda_": 1e-8,
        "line_search_method": "strict_decrease",
        "verbose": True,
        "line_search": True,
        "ls_beta": 0.5,
        # "ls_min_step": 1e-8,
        "ls_max_steps": 20,
    }
elif _FORWARD_SOLVER == "scipy_minimize":
    _FORWARD_OPTIONS = {"max_iters": 120, "tol": 1e-8}
elif _FORWARD_SOLVER == "cyipopt":
    _FORWARD_OPTIONS = {
        "max_iters": 120,
        "tol": 1e-8,
        # solve(..., solver="cyipopt") forwards unknown keys to IPOPT backend options.
        "print_level": 5,
        "print_timing_statistics": "yes",
    }
else:
    raise ValueError(f"Unsupported _FORWARD_SOLVER={_FORWARD_SOLVER!r}")
_ACTIVE_MODE = "gradient"
_IOC_MAX_ITERS = 10000
_IKKT_TOL = 1e-6


def _normalize_simplex(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=float).reshape(-1)
    if x.size == 0:
        return x
    x = np.maximum(x, 0.0)
    s = float(x.sum())
    if s <= 0.0:
        return np.full(x.shape, 1.0 / float(x.size), dtype=float)
    return x / s


def main() -> int:
    if not _MODEL_PATH.is_file():
        raise SystemExit(f"Model file not found: {_MODEL_PATH}")
    if not _DSL_PATH.is_file():
        raise SystemExit(f"DSL file not found: {_DSL_PATH}")

    dsl = load_problem_toml(_DSL_PATH)

    # kots = Kots.from_json_file(str(_MODEL_PATH), order=_ORDER)
    kots = Kots.from_urdf_file(str(_MODEL_PATH), order=_ORDER)

    # 1) Forward optimization (solve in nullspace-reduced coordinates)
    compiled_fwd = compile_kots_trajectory_problem(
        dsl,
        model=kots,
        data=kots.state_dict_,
    )
    runtime_fwd_full = compiled_fwd.runtime
    reduction_fwd = build_nullspace_equality_reduction(
        runtime_fwd_full,
        eq_selector_attr=_NULLSPACE_EQ_SELECTOR_ATTR,
    )
    runtime_fwd = reduction_fwd.runtime

    out_fwd = solve(
        runtime_fwd,
        solver=_FORWARD_SOLVER,
        options=_FORWARD_OPTIONS,
    )
    z_star = np.asarray(out_fwd.solution, dtype=float).reshape(-1)
    x_star_full = np.asarray(reduction_fwd.lift(z_star), dtype=float).reshape(-1)

    runtime_fwd_full.pack.apply_dx(
        x_star_full - np.asarray(runtime_fwd_full.pack.get(), dtype=float).reshape(-1)
    )
    kkt = check_kkt_conditions(
        runtime_fwd_full,
        stationarity_tol=1e-6,
        eq_tol=1e-6,
        ineq_tol=1e-8,
    )

    # Extract "true" objective weights from forward runtime term costs (reduced coordinates).
    src_true = RuntimeStationaritySource(runtime_fwd)
    src_true.set_point(z_star)
    contrib_true = filter_stationarity_contributions(
        src_true.term_contributions(required=src_true.required_list(None)),
        include_constraints=False,
    )
    w_true = _normalize_simplex(
        np.asarray(
            [0.0 if c.reference_weight is None else float(c.reference_weight) for c in contrib_true],
            dtype=float,
        )
    )

    # 2) Inverse stage: set objective weights = 1, then infer simplex weights (reduced coordinates).
    runtime_inv_full = compile_kots_trajectory_problem(
        dsl,
        model=kots,
        data=kots.state_dict_,
    ).runtime

    constraint_indices = set(int(i) for i in runtime_inv_full.find_constraint_term_indices())
    for i in range(len(runtime_inv_full.problem.terms)):
        if i not in constraint_indices:
            runtime_inv_full.set_cost_weight(i, 1.0)

    reduction_inv = build_nullspace_equality_reduction(
        runtime_inv_full,
        eq_selector_attr=_NULLSPACE_EQ_SELECTOR_ATTR,
    )
    runtime_inv = reduction_inv.runtime
    z_star_inv = np.asarray(reduction_inv.project(x_star_full), dtype=float).reshape(-1)

    src_inv = RuntimeStationaritySource(runtime_inv)
    src_inv.set_point(z_star_inv)
    contrib_inv = filter_stationarity_contributions(
        src_inv.term_contributions(required=src_inv.required_list(None)),
        include_constraints=False,
    )

    if tuple(c.term_index for c in contrib_true) != tuple(c.term_index for c in contrib_inv):
        raise RuntimeError("objective term order mismatch between forward and inverse runtimes")

    A_col, term_indices = build_stationarity_gradient_matrix(contrib_inv, n_total=int(src_inv.n_total))
    active_idx, active_grad_idx, active_res_idx = select_active_stationarity_indices(
        contrib_inv,
        mode=_ACTIVE_MODE,
    )

    w_hat = np.zeros((len(contrib_inv),), dtype=float)
    ikkt_residual = float("nan")
    simplex_out = None
    ioc_identifiable = len(active_idx) > 0
    if ioc_identifiable:
        A_active = np.asarray(A_col[:, active_idx], dtype=float)
        simplex_out = solve_simplex_min_norm(
            A_active,
            method="qr_nullspace",
            max_iters=int(_IOC_MAX_ITERS),
            tol=1e-12,
        )
        w_hat[np.asarray(active_idx, dtype=int)] = np.asarray(simplex_out.solution, dtype=float).reshape(-1)

    w_hat = _normalize_simplex(w_hat)
    if ioc_identifiable:
        ikkt_residual = float(np.linalg.norm(np.asarray(A_col[:, active_idx], dtype=float) @ w_hat[np.asarray(active_idx, dtype=int)]))
    ikkt_ok = bool(ioc_identifiable and ikkt_residual <= float(_IKKT_TOL))

    # 3) Report
    print("=== 11_forward_then_inverse_ioc ===")
    print(f"model={_MODEL_PATH} order={_ORDER}")
    print(f"dsl={_DSL_PATH}")
    stats_fwd = out_fwd.stats
    print(
        f"forward: solver={_FORWARD_SOLVER} nullspace_eq=True "
        f"status={stats_fwd.status} converged={stats_fwd.converged} iters={stats_fwd.iterations} "
        f"cost0={float(stats_fwd.initial_objective or 0.0):.3e} "
        f"cost={float(stats_fwd.objective or 0.0):.3e} "
        f"rnorm={float(stats_fwd.residual_norm or 0.0):.3e} "
        f"dxnorm={float(stats_fwd.step_norm or 0.0):.3e}"
    )

    print(
        format_ioc_report(
            active_mode=_ACTIVE_MODE,
            active_idx=list(active_idx),
            active_grad_idx=list(active_grad_idx),
            active_res_idx=list(active_res_idx),
            term_indices=list(term_indices),
            w_true=w_true,
            w_hat=w_hat,
            ioc_identifiable=ioc_identifiable,
            ikkt_ok=ikkt_ok,
            ikkt_residual=ikkt_residual,
            ikkt_tol=_IKKT_TOL,
            kkt=kkt,
            simplex_out=simplex_out,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
