from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

try:
    from robokots.kots import Kots
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires RoboKots with compatible dependencies.\n"
        "Install `robokots` and ensure `mathrobo` provides CMVector.\n"
        "Then run:\n"
        "  PYTHONPATH=. python examples/main_robokots_traj_dynamics.py"
    ) from e

from eiopt.optimize.ioc import (
    format_ioc_report,
    prepare_ioc_weights,
)
from eiopt.optimize.builder import load_problem_toml
from eiopt.optimize.dsl import split_terms_by_component
from eiopt.optimize.reductions import build_nullspace_equality_reduction
from eiopt.optimize.report import format_solve_report
from eiopt.optimize.solvers import solve
from eiopt.optimize_backends.kots import compile_kots_trajectory_problem

_EXAMPLES_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _EXAMPLES_DIR / "model" / "sample_robot.json"
if not _MODEL_PATH.is_file():
    _MODEL_PATH = _EXAMPLES_DIR / "examples" / "models" / "planar2.json"
_ORDER = 5
_DSL_PATH = _EXAMPLES_DIR / "task_robot_doc_ioc.toml"
if not _DSL_PATH.is_file():
    _DSL_PATH = _EXAMPLES_DIR / "examples" / "dsl" / "robokots_traj_dynamics_d12.toml"

# Solver selection (edit these in code)
_SOLVER = "gauss_newton"  # "gauss_newton" | "scipy_minimize" | "cyipopt"
_SCIPY_METHOD = "L-BFGS-B"
_SCIPY_OPTIONS = {"maxiter": 1000}
_IPOPT_OPTIONS = {"max_iter": 2000,
                  "tol": 1e-6, "acceptable_tol": 1e-5, "acceptable_iter": 10,
                  "print_level": 5, "print_timing_statistics": "yes"}
_USE_NULLSPACE_EQ = True
_RUN_IOC_PREP = True
_IOC_WEIGHT_TOL = 1e-10
_IOC_ACTIVE_GRAD_TOL = 1e-10
_IOC_VERBOSE_SINGULAR_VALUES = True
_IOC_VERBOSE_TERM_DETAILS = True
_SPLIT_TERMS_BY_JOINT = False

def run_trajectory_dynamics_demo(
    *,
    solver: str = _SOLVER,
    max_iters: int = 1000,
    scipy_method: str = _SCIPY_METHOD,
    scipy_options: dict | None = None,
    ipopt_options: dict | None = None,
) -> int:
    if not _MODEL_PATH.is_file():
        raise SystemExit(f"Model file not found: {_MODEL_PATH}")
    if not _DSL_PATH.is_file():
        raise SystemExit(f"DSL file not found: {_DSL_PATH}")

    kots = Kots.from_json_file(str(_MODEL_PATH), order=_ORDER)
    data = kots.state_dict_
    dsl = load_problem_toml(_DSL_PATH)
    scipy_options_use = _SCIPY_OPTIONS if scipy_options is None else scipy_options
    ipopt_options_use = _IPOPT_OPTIONS if ipopt_options is None else ipopt_options

    compiled = compile_kots_trajectory_problem(dsl, model=kots, data=data)
    if _SPLIT_TERMS_BY_JOINT:
        q_dim = int(compiled.trajectory_map.q_dim)
        term_linear = compiled.runtime.linearize_terms(weighted=False)
        split_term_indices = [
            i
            for i, term in enumerate(term_linear)
            if int(np.asarray(term.residual, dtype=float).size) > 1
            and int(np.asarray(term.residual, dtype=float).size) % q_dim == 0
        ]
        if len(split_term_indices) > 0:
            dsl_split = copy.deepcopy(dsl)
            split_terms_by_component(
                dsl_split,
                segment_dim=q_dim,
                term_indices=split_term_indices,
            )
            compiled = compile_kots_trajectory_problem(dsl_split, model=kots, data=data)
            print(
                "Per-joint objective expansion: "
                f"q_dim={q_dim}, expanded_terms={len(split_term_indices)}, "
                f"total_terms={len(compiled.runtime.problem.terms)}"
            )
    runtime = compiled.runtime
    traj_map = compiled.trajectory_map

    dt = float(compiled.dt)
    x_initial = np.asarray(runtime.pack.get(), dtype=float).reshape(-1).copy()

    # Stage 1: solve once in full space (without nullspace reduction).
    x_stage1, cost0_stage1, cost_stage1, iters_stage1, rnorm_stage1, dxnorm_stage1, converged_stage1 = solve(
        runtime,
        solver=solver,
        max_iters=max_iters,
        scipy_method=scipy_method,
        scipy_options=scipy_options_use,
        ipopt_options=ipopt_options_use,
    )

    # Stage 2: only when nullspace reduction is enabled.
    nullspace_eq = build_nullspace_equality_reduction(runtime) if _USE_NULLSPACE_EQ else None
    runtime_for_solve = runtime if nullspace_eq is None else nullspace_eq.runtime
    if nullspace_eq is None:
        x0_report = x_initial
        x_star_solve = np.asarray(x_stage1, dtype=float).reshape(-1).copy()
        x_star = x_star_solve.copy()
        _cost0 = cost0_stage1
        _cost = cost_stage1
        _iters = iters_stage1
        _rnorm = rnorm_stage1
        _dxnorm = dxnorm_stage1
        _converged = converged_stage1
    else:
        x0_warm = np.asarray(x_stage1, dtype=float).reshape(-1).copy()
        x0_solve = nullspace_eq.project(x0_warm)
        x_cur_solve = np.asarray(runtime_for_solve.pack.get(), dtype=float).reshape(-1)
        runtime_for_solve.pack.apply_dx(np.asarray(x0_solve, dtype=float).reshape(-1) - x_cur_solve)

        x_star_solve, _cost0, _cost, _iters, _rnorm, _dxnorm, _converged = solve(
            runtime_for_solve,
            solver=solver,
            max_iters=max_iters,
            scipy_method=scipy_method,
            scipy_options=scipy_options_use,
            ipopt_options=ipopt_options_use,
        )
        x_star = nullspace_eq.lift(x_star_solve)
        x0_report = x0_warm

    steps = int(traj_map.steps)
    if nullspace_eq is not None:
        print(
            "Warmup (full-space solve): "
            f"converged={converged_stage1} "
            f"iters={iters_stage1} "
            f"cost0={cost0_stage1:.3e} "
            f"cost={cost_stage1:.3e} "
            f"rnorm={rnorm_stage1:.3e} "
            f"dxnorm={dxnorm_stage1:.3e}"
        )
    print(
        format_solve_report(
            runtime,
            x0=x0_report,
            x_star=x_star,
            include_named=False,
            solve_summary={
                "converged": _converged,
                "iters": _iters,
                "cost0": _cost0,
                "cost": _cost,
                "rnorm": _rnorm,
                "dxnorm": _dxnorm,
            },
            trajectory_summary={
                "steps": steps,
                "dt": dt,
                "p_dim": traj_map.p_dim,
                "dynamics_fields": compiled.dynamics_fields,
            },
        )
    )
    if _RUN_IOC_PREP:
        ioc = prepare_ioc_weights(
            runtime_for_solve,
            x_opt=x_star_solve,
            active_grad_tol=_IOC_ACTIVE_GRAD_TOL,
            weight_tol=_IOC_WEIGHT_TOL,
            max_iters=50000,
        )
        print(
            format_ioc_report(
                ioc,
                include_singular_values=_IOC_VERBOSE_SINGULAR_VALUES,
                include_term_details=_IOC_VERBOSE_TERM_DETAILS,
            )
        )
    return 0


def main() -> int:
    return run_trajectory_dynamics_demo(
        solver=_SOLVER,
        max_iters=1000,
        scipy_method=_SCIPY_METHOD,
        scipy_options=_SCIPY_OPTIONS,
        ipopt_options=_IPOPT_OPTIONS,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
