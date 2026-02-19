from __future__ import annotations

import argparse
from pathlib import Path

from eiopt.optimize.builder import load_problem_toml
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
    parser = argparse.ArgumentParser(description="RoboKots trajectory dynamics example.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot series declared in term.attrs.plot.",
    )
    args = parser.parse_args()

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
    x_star, initial_cost, cost, iters, rnorm, dxnorm, converged = solve(
        runtime,
        solver="gauss_newton",
        options={"max_iters": 500, "tol_dx": 1e-8},
    )

    steps = int(compiled.trajectory_map.steps)

    print("=== 07_robokots_trajectory_dynamics ===")
    print(f"dsl={_DSL_PATH}")
    print(f"model={_MODEL_PATH} (order={_ORDER})")
    print(
        format_solve_report(
            runtime,
            x0=x0,
            x_star=x_star,
            include_kkt=True,
            kkt_kwargs={"stationarity_tol": 1e-6},
            solve_summary={
                "converged": converged,
                "iters": iters,
                "cost0": initial_cost,
                "cost": cost,
                "rnorm": rnorm,
                "dxnorm": dxnorm,
            },
            trajectory_summary={
                "steps": steps,
                "dt": compiled.dt,
                "p_dim": compiled.trajectory_map.p_dim,
                "dynamics_fields": compiled.dynamics_fields,
            },
        )
    )

    if args.plot:
        from eiopt.optimize.plot import plot_term_attrs

        fig, _ax, series = plot_term_attrs(
            runtime,
            title="07_robokots_trajectory_dynamics",
        )
        del fig
        print(f"plotted series={len(series)} from term.attrs.plot")
        import matplotlib.pyplot as plt

        plt.savefig("trajectory.png", dpi=200)


if __name__ == "__main__":
    main()
