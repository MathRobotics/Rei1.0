from __future__ import annotations

import argparse
from pathlib import Path

from rei import format_solve_report, load_problem_spec_toml, solve
from rei.optimize_backends.pinocchio import compile_pinocchio_trajectory_problem

try:
    import pinocchio as pin
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires the robotics Pinocchio bindings.\n"
        "Install dependencies (e.g. `uv sync --group pinocchio`) and re-run:\n"
        "  PYTHONPATH=. python examples/pinocchio_trajectory_dynamics.py"
    ) from e

_EXAMPLES_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _EXAMPLES_DIR / "models" / "planar2.urdf"
_SPEC_PATH = _EXAMPLES_DIR / "spec" / "pinocchio_traj_dynamics.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinocchio trajectory dynamics example.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot series declared in term.attrs.plot.",
    )
    args = parser.parse_args()

    if not _MODEL_PATH.is_file():
        raise SystemExit(f"Model file not found: {_MODEL_PATH}")
    if not _SPEC_PATH.is_file():
        raise SystemExit(f"TOML spec file not found: {_SPEC_PATH}")

    problem = load_problem_spec_toml(_SPEC_PATH)
    model = pin.buildModelFromUrdf(str(_MODEL_PATH))
    data = model.createData()

    compiled = compile_pinocchio_trajectory_problem(
        problem,
        model=model,
        data=data,
    )
    runtime = compiled.runtime

    x0 = runtime.pack.get().copy()
    out = solve(
        runtime,
        solver="gauss_newton",
        options={"max_iters": 400, "tol_dx": 1e-8},
    )
    steps = int(compiled.trajectory_map.steps)

    print("=== pinocchio_trajectory_dynamics ===")
    print(f"spec={_SPEC_PATH}")
    print(f"model={_MODEL_PATH}")
    print(
        format_solve_report(
            runtime,
            x0=x0,
            outcome=out,
            include_kkt=True,
            trajectory_summary={
                "steps": steps,
                "dt": compiled.dt,
                "p_dim": compiled.trajectory_map.p_dim,
                "dynamics_fields": compiled.dynamics_fields,
            },
        )
    )

    if args.plot:
        from rei.optimize.plot import plot_term_attrs

        fig, _ax, series = plot_term_attrs(
            runtime,
            title="pinocchio_trajectory_dynamics",
        )
        del fig
        print(f"plotted series={len(series)} from term.attrs.plot")
        import matplotlib.pyplot as plt

        plt.savefig("trajectory.png", dpi=200)


if __name__ == "__main__":
    main()
