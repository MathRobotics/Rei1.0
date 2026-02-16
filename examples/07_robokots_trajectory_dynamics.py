from __future__ import annotations

from pathlib import Path

import numpy as np

from eiopt.optimize.builder import load_problem_toml
from eiopt.optimize.dsl import build_trajectory_map_with_derivative
from eiopt.optimize.report import format_solve_report
from eiopt.optimize.solvers import solve
from eiopt.optimize_backends.kots import compile_kots_trajectory_problem

try:
    from robokots.kots import Kots
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires RoboKots.\n"
        "Install dependencies (e.g. `uv sync --group kots`) and re-run:\n"
        "  PYTHONPATH=. python examples/07_robokots_trajectory_dynamics.py"
    ) from e

_EXAMPLES_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _EXAMPLES_DIR / "models" / "planar2.json"
_DSL_PATH = _EXAMPLES_DIR / "dsl" / "robokots_traj_dynamics_d12.toml"
_ORDER = 5


def main() -> None:
    if not _MODEL_PATH.is_file():
        raise SystemExit(f"Model file not found: {_MODEL_PATH}")
    if not _DSL_PATH.is_file():
        raise SystemExit(f"DSL file not found: {_DSL_PATH}")

    dsl = load_problem_toml(_DSL_PATH)
    kots = Kots.from_json_file(str(_MODEL_PATH), order=_ORDER)

    compiled = compile_kots_trajectory_problem(
        dsl,
        model=kots,
        data=kots.state_dict_,
    )
    runtime = compiled.runtime

    x0 = runtime.pack.get().copy()
    x_star, cost, iters, rnorm, dxnorm, converged = solve(
        runtime,
        solver="gauss_newton",
        max_iters=500,
        tol_dx=1e-8,
    )
    runtime.pack.apply_dx(x_star - runtime.pack.get().copy())

    steps = int(compiled.trajectory_map.steps)
    q_traj = np.vstack([compiled.trajectory_map.q_at(x_star, k) for k in range(steps)])
    traj_d1 = build_trajectory_map_with_derivative(
        dsl["trajectory"],
        derivative_order=1,
        derivative_wrt="time",
        default_steps=steps,
        default_q_dim=int(compiled.trajectory_map.q_dim),
        default_dt=float(compiled.dt),
    )
    qdot0 = np.asarray(traj_d1.q_at(x_star, 0), dtype=float).reshape(-1)
    qdotT = np.asarray(traj_d1.q_at(x_star, steps - 1), dtype=float).reshape(-1)

    tau = runtime.collect_state_traj(
        owner_type="total_joint",
        owner_name="robot",
        dtype="dynamics",
        field="torque",
        ks=range(steps),
    )
    tau_d1 = runtime.collect_state_traj(
        owner_type="total_joint",
        owner_name="robot",
        dtype="dynamics",
        field="torque_d1",
        ks=range(steps),
    )
    tau_d2 = runtime.collect_state_traj(
        owner_type="total_joint",
        owner_name="robot",
        dtype="dynamics",
        field="torque_d2",
        ks=range(steps),
    )

    print("=== 07_robokots_trajectory_dynamics ===")
    print(f"dsl={_DSL_PATH}")
    print(f"model={_MODEL_PATH} (order={_ORDER})")
    print(f"converged={converged} iters={iters} cost={cost:.3e} rnorm={rnorm:.3e} dxnorm={dxnorm:.3e}")
    print(f"steps={steps} dt={compiled.dt:g} p_dim={compiled.trajectory_map.p_dim} dynamics_fields={compiled.dynamics_fields}")
    print(f"q(0)={q_traj[0]}")
    print(f"q(T)={q_traj[-1]}")
    print(f"dq/dt(0)={qdot0}")
    print(f"dq/dt(T)={qdotT}")
    print(f"max|torque|={float(np.max(np.abs(tau))):.3e}")
    print(f"max|torque_d1|={float(np.max(np.abs(tau_d1))):.3e}")
    print(f"max|torque_d2|={float(np.max(np.abs(tau_d2))):.3e}")
    print(format_solve_report(runtime, x0=x0, x_star=x_star))


if __name__ == "__main__":
    main()
