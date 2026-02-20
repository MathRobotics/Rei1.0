from __future__ import annotations

from pathlib import Path

from eiopt.optimize.builder import compile_nls_problem, load_problem_toml
from eiopt.optimize.report import format_solve_report
from eiopt.optimize.solvers import solve

try:
    import pinocchio as pin
    from eiopt.backends.state.robotics.pinocchio import PinocchioStateBuilder
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires the robotics Pinocchio bindings.\n"
        "Install dependencies (e.g. `uv sync --group pinocchio`) and re-run:\n"
        "  PYTHONPATH=. python examples/04_pinocchio_ik.py"
    ) from e

_EXAMPLES_DIR = Path(__file__).resolve().parent
_DSL_PATH = _EXAMPLES_DIR / "dsl" / "ik_pos.toml"
_URDF_PATH = _EXAMPLES_DIR / "models" / "planar2.urdf"


def main() -> None:
    if not _DSL_PATH.is_file():
        raise SystemExit(f"DSL file not found: {_DSL_PATH}")
    if not _URDF_PATH.is_file():
        raise SystemExit(f"URDF file not found: {_URDF_PATH}")

    model = pin.buildModelFromUrdf(str(_URDF_PATH))
    data = model.createData()
    dsl = load_problem_toml(_DSL_PATH)

    builder = PinocchioStateBuilder(model, data, q_var="q", fields=("pos",))
    runtime = compile_nls_problem(dsl, build_state=builder.build_state)

    x0 = runtime.pack.get().copy()
    out = solve(
        runtime,
        solver="gauss_newton",
        options={"max_iters": 200, "tol_dx": 1e-8},
    )
    x_star = out.solution
    stats = out.stats

    print("=== 04_pinocchio_ik ===")
    print(f"dsl={_DSL_PATH}")
    print(f"model={_URDF_PATH}")
    print(
        f"status={stats.status} converged={stats.converged} iters={stats.iterations} "
        f"cost0={float(stats.initial_objective or 0.0):.3e} "
        f"cost={float(stats.objective or 0.0):.3e} "
        f"rnorm={float(stats.residual_norm or 0.0):.3e} "
        f"dxnorm={float(stats.step_norm or 0.0):.3e}"
    )
    print(f"x0={x0}")
    print(f"x*={x_star}")
    print(format_solve_report(runtime, x0=x0, outcome=out))


if __name__ == "__main__":
    main()
