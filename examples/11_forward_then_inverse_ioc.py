from __future__ import annotations

from pathlib import Path

import numpy as np

from eiopt import (
    RuntimeStationaritySource,
    build_stationarity_gradient_matrix,
    format_timing_report,
    filter_stationarity_contributions,
    select_active_stationarity_indices,
    solve_simplex_min_norm,
)
from eiopt.optimize.builder import load_problem_toml
from eiopt.optimize.kkt import check_kkt_conditions
from eiopt.optimize.reductions import build_nullspace_equality_reduction
from eiopt.optimize.solvers import solve
from eiopt.optimize_backends.kots import compile_kots_trajectory_problem

try:
    from robokots.kots import Kots
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires RoboKots.\n"
        "Install dependencies (e.g. `uv sync --group kots`) and re-run:\n"
        "  PYTHONPATH=. python examples/11_forward_then_inverse_ioc.py"
    ) from e

_EXAMPLES_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _EXAMPLES_DIR / "models" / "planar2.json"
_DSL_PATH = _EXAMPLES_DIR / "dsl" / "robokots_traj_dynamics_d12.toml"  # up to torque_d1
_ORDER = 5

# Simple fixed settings (edit directly if needed)
_FORWARD_SOLVER = "gauss_newton"
_FORWARD_OPTIONS = {"max_iters": 120, "tol_dx": 1e-8}
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

    kots = Kots.from_json_file(str(_MODEL_PATH), order=_ORDER)

    # 1) Forward optimization (solve in nullspace-reduced coordinates)
    runtime_fwd_full = compile_kots_trajectory_problem(
        dsl,
        model=kots,
        data=kots.state_dict_,
    ).runtime
    reduction_fwd = build_nullspace_equality_reduction(runtime_fwd_full)
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

    reduction_inv = build_nullspace_equality_reduction(runtime_inv_full)
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
    ikkt_residual = 0.0
    simplex_out = None
    if len(active_idx) > 0:
        A_active = np.asarray(A_col[:, active_idx], dtype=float)
        simplex_out = solve_simplex_min_norm(
            A_active,
            method="qr_nullspace",
            max_iters=int(_IOC_MAX_ITERS),
            tol=1e-12,
        )
        w_hat[np.asarray(active_idx, dtype=int)] = np.asarray(simplex_out.solution, dtype=float).reshape(-1)

    w_hat = _normalize_simplex(w_hat)
    if len(active_idx) > 0:
        ikkt_residual = float(np.linalg.norm(np.asarray(A_col[:, active_idx], dtype=float) @ w_hat[np.asarray(active_idx, dtype=int)]))
    ikkt_ok = bool(ikkt_residual <= float(_IKKT_TOL))

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
    print(format_timing_report(out_fwd.timing, title="forward timing"))
    print(
        f"KKT: ok={kkt.ok} stationarity_inf={kkt.stationarity_inf:.3e} "
        f"eq_violation_inf={kkt.eq_violation_inf:.3e} ineq_violation_inf={kkt.ineq_violation_inf:.3e}"
    )
    print(
        f"iKKT(active stationarity): ok={ikkt_ok} residual_norm={ikkt_residual:.3e} tol={_IKKT_TOL:.3e}"
    )
    print(f"active local idx (selected)={list(active_idx)}")
    print(f"active local idx (gradient)={list(active_grad_idx)}")
    print(f"active local idx (residual)={list(active_res_idx)}")
    print(f"term_indices={list(term_indices)}")
    print(f"w_true={w_true}")
    print(f"w_hat={w_hat}")
    print(f"L1 error={float(np.linalg.norm(w_hat - w_true, ord=1)):.3e}")
    if simplex_out is not None:
        simplex_stats = simplex_out.stats
        print(
            "simplex: "
            f"status={simplex_stats.status} "
            f"converged={simplex_stats.converged} "
            f"iters={simplex_stats.iterations} "
            f"objective={float(simplex_stats.objective or float('nan')):.3e}"
        )
        print(format_timing_report(simplex_out.timing, title="simplex timing"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
