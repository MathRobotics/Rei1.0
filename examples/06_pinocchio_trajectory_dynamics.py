from __future__ import annotations

from pathlib import Path

import numpy as np

from eiopt.optimize.builder import load_problem_toml
from eiopt.optimize.dsl import build_trajectory_map_with_derivative
from eiopt.optimize.report import format_solve_report
from eiopt.optimize.solvers import solve
from eiopt.optimize_backends.pinocchio import compile_pinocchio_trajectory_problem

try:
    import pinocchio as pin
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires the robotics Pinocchio bindings.\n"
        "Install dependencies (e.g. `uv sync --group pinocchio`) and re-run:\n"
        "  PYTHONPATH=. python examples/06_pinocchio_trajectory_dynamics.py"
    ) from e

_EXAMPLES_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _EXAMPLES_DIR / "models" / "planar2.urdf"
_DSL_PATH = _EXAMPLES_DIR / "dsl" / "pinocchio_traj_dynamics.toml"


def main() -> None:
    if not _MODEL_PATH.is_file():
        raise SystemExit(f"Model file not found: {_MODEL_PATH}")
    if not _DSL_PATH.is_file():
        raise SystemExit(f"DSL file not found: {_DSL_PATH}")

    dsl = load_problem_toml(_DSL_PATH)
    model = pin.buildModelFromUrdf(str(_MODEL_PATH))
    data = model.createData()

    compiled = compile_pinocchio_trajectory_problem(
        dsl,
        model=model,
        data=data,
    )
    runtime = compiled.runtime

    x0 = runtime.pack.get().copy()
    x_star, cost, iters, rnorm, dxnorm, converged = solve(
        runtime,
        solver="gauss_newton",
        max_iters=400,
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
    tau_traj = runtime.collect_state_traj(
        owner_type="total_joint",
        owner_name="robot",
        dtype="dynamics",
        field="torque",
        ks=range(steps),
    )

    print("=== 06_pinocchio_trajectory_dynamics ===")
    print(f"dsl={_DSL_PATH}")
    print(f"model={_MODEL_PATH}")
    print(f"converged={converged} iters={iters} cost={cost:.3e} rnorm={rnorm:.3e} dxnorm={dxnorm:.3e}")
    print(f"steps={steps} dt={compiled.dt:g} p_dim={compiled.trajectory_map.p_dim} dynamics_fields={compiled.dynamics_fields}")
    print(f"q(0)={q_traj[0]}")
    print(f"q(T)={q_traj[-1]}")
    print(f"dq/dt(0)={qdot0}")
    print(f"dq/dt(T)={qdotT}")
    print(f"max|torque|={float(np.max(np.abs(tau_traj))):.3e}")
    print(format_solve_report(runtime, x0=x0, x_star=x_star))


if __name__ == "__main__":
    main()
