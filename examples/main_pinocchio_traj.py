from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import pinocchio as pin  # robotics library
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires the robotics `pinocchio` Python bindings.\n"
        "Install Pinocchio in your environment (e.g. via conda-forge) and re-run:\n"
        "  PYTHONPATH=. python examples/main_pinocchio_traj.py"
    ) from e

from eiopt import format_solve_report, load_problem_toml, solve_runtime
from eiopt.backends.pinocchio import compile_pinocchio_trajectory_problem

_EXAMPLES_DIR = Path(__file__).resolve().parent
_URDF_PATH = _EXAMPLES_DIR / "models" / "planar2.urdf"
_DSL_PATH = _EXAMPLES_DIR / "dsl" / "pinocchio_traj_joint.toml"


def main() -> int:
    if not _URDF_PATH.is_file():
        raise SystemExit(f"URDF file not found: {_URDF_PATH}")
    if not _DSL_PATH.is_file():
        raise SystemExit(f"DSL file not found: {_DSL_PATH}")

    model = pin.buildModelFromUrdf(str(_URDF_PATH))
    data = model.createData()
    dsl = load_problem_toml(_DSL_PATH)

    compiled = compile_pinocchio_trajectory_problem(
        dsl,
        model=model,
        data=data,
    )
    runtime = compiled.runtime
    traj_map = compiled.trajectory_map

    x0 = runtime.pack.get().copy()
    x_star, cost, iters, rnorm, dxnorm, converged = solve_runtime(
        runtime,
        solver="gauss_newton",
        max_iters=30,
    )

    steps = int(traj_map.steps)
    q_opt = np.vstack([traj_map.q_at(x_star, k) for k in range(steps)])
    eq_terms = runtime.linearize_constraint_terms(kind="eq", weighted=False)
    eq_residual_norm = (
        float(np.linalg.norm(np.concatenate([t.residual for t in eq_terms], axis=0)))
        if len(eq_terms) > 0
        else 0.0
    )

    print("Optimization completed.")
    print("\tIterations:", iters)
    print("\tFinal cost:", cost)
    print("\tFinal residual norm:", rnorm)
    print("\tFinal step norm:", dxnorm)
    print("\tConverged:", converged)
    print("\tEq residual norm:", eq_residual_norm)
    print("p*:", x_star)
    for k in range(steps):
        print(f"q[{k}] =", q_opt[k])
    print(format_solve_report(runtime, x0=x0, x_star=x_star))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
