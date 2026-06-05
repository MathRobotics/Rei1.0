from __future__ import annotations

import argparse
from pathlib import Path

from rei import SolveOutcome, format_solve_report, load_problem_spec_toml, solve
from rei.optimize_backends.kots import compile_kots_trajectory_problem
from rei.optimize.reductions import build_nullspace_equality_reduction

try:
    from robokots.kots import Kots
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires RoboKots.\n"
        "Install dependencies (e.g. `uv sync --group kots`) and re-run:\n"
        "  PYTHONPATH=. python examples/robokots_trajectory_dynamics.py"
    ) from e

_EXAMPLES_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _EXAMPLES_DIR / "models" / "planar2.json"
_SPEC_PATH = _EXAMPLES_DIR / "spec" / "robokots_traj_dynamics_d12.toml"
_ORDER = 5


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboKots trajectory dynamics example.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot series declared by term plot metadata.",
    )
    args = parser.parse_args()

    if not _MODEL_PATH.is_file():
        raise SystemExit(f"Model file not found: {_MODEL_PATH}")
    if not _SPEC_PATH.is_file():
        raise SystemExit(f"TOML spec file not found: {_SPEC_PATH}")

    problem = load_problem_spec_toml(_SPEC_PATH)
    kots = Kots.from_json_file(str(_MODEL_PATH), order=_ORDER)

    compiled = compile_kots_trajectory_problem(
        problem,
        model=kots,
        data=kots.state_dict_,
    )
    runtime = compiled.runtime

    x0 = runtime.pack.get().copy()
    reduction = build_nullspace_equality_reduction(
        runtime,
        eq_selector_attr="enforce",
        eq_selector_value="nullspace",
    )
    out_reduced = solve(
        reduction.runtime,
        solver="gauss_newton",
        options={"max_iters": 500, "tol_dx": 1e-8},
    )
    reduction.runtime.update_state_if_needed()
    out = SolveOutcome(
        solution=runtime.pack.get().copy(),
        stats=out_reduced.stats,
        timing=out_reduced.timing,
        meta={
            **out_reduced.meta,
            "x0": x0,
            "reduced_solution": out_reduced.solution.copy(),
        },
    )
    steps = int(compiled.trajectory_map.steps)

    print("=== robokots_trajectory_dynamics ===")
    print(f"spec={_SPEC_PATH}")
    print(f"model={_MODEL_PATH} (order={_ORDER})")
    print(
        "nullspace="
        f"rank={reduction.rank} "
        f"dim={reduction.runtime.pack.n_total} "
        f"feasibility={reduction.feasibility_residual_norm:.3e}"
    )
    print(
        format_solve_report(
            runtime,
            x0=x0,
            outcome=out,
            include_kkt=True,
            kkt_kwargs={"stationarity_tol": 1e-6},
            trajectory_summary={
                "steps": steps,
                "dt": compiled.dt,
                "p_dim": compiled.trajectory_map.p_dim,
                "dynamics_fields": compiled.dynamics_fields,
            },
        )
    )

    if args.plot:
        from rei.optimize.plot import collect_plot_series_from_compiled_term_attrs, plot_series

        series = collect_plot_series_from_compiled_term_attrs(compiled)
        fig, _ax, series = plot_series(
            series,
            title="robokots_trajectory_dynamics",
        )
        del fig
        print(f"plotted series={len(series)} from term plot metadata")
        import matplotlib.pyplot as plt

        plt.savefig("trajectory.png", dpi=200)


if __name__ == "__main__":
    main()
