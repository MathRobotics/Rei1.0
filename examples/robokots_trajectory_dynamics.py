from __future__ import annotations

import argparse
from pathlib import Path

from rei import SolveOutcome, format_solve_report, load_problem_spec_toml, solve
from rei.optimize_backends.kots import compile_kots_trajectory_problem
from rei.optimize.reductions import build_nullspace_equality_reduction
from rei.optimize.history import format_solver_history

_EXAMPLES_DIR = Path(__file__).resolve().parent
_SPEC_PATH = _EXAMPLES_DIR / "spec" / "robokots_traj_dynamics_d12.toml"
_KOTS_BACKEND = "rust"


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboKots trajectory dynamics example.")
    parser.add_argument("--jacobian-method", choices=("analytic", "numerical", "autodiff"), default="analytic")
    parser.add_argument("--jacobian-strategy", choices=("dense", "mul"), default="mul")
    parser.add_argument("--batch-trajectory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--numerical-eps", type=float, default=1e-8)
    parser.add_argument("--autodiff-mode", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--autodiff-jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--history", type=Path, help="Stream history to a new JSONL file.")
    parser.add_argument("--solver", choices=("levenberg_marquardt", "gauss_newton", "gauss_newton_operator", "gauss_newton_krylov", "lm-ls"), default="levenberg_marquardt")
    parser.add_argument("--trial-history", type=Path, help="LM or Krylov trust-region trial JSONL path.")
    parser.add_argument("--show-trials", action="store_true", help="Show LM or Krylov trust-region trials.")
    parser.add_argument("--history-vectors", action="store_true", help="Include reduced variables and Jᵀr vectors.")
    parser.add_argument("--show-history", action="store_true", help="Show iterations (already enabled by default).")
    parser.add_argument("--quiet", action="store_true", help="Disable live iteration output.")
    parser.add_argument("--line-search-history", type=Path, help="Separate line-search JSONL path (default: <history stem>.line_search.jsonl).")
    parser.add_argument("--show-line-search", action="store_true", help="Show every line-search trial.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot series declared by term plot metadata.",
    )
    args = parser.parse_args()
    trust_solver = args.solver in ("levenberg_marquardt", "gauss_newton_krylov")
    if trust_solver and (args.line_search_history or args.show_line_search):
        parser.error("--line-search-history/--show-line-search require --solver gauss_newton, gauss_newton_operator or lm-ls")
    if not trust_solver and (args.trial_history or args.show_trials):
        parser.error("--trial-history/--show-trials require --solver levenberg_marquardt or gauss_newton_krylov")

    if not _SPEC_PATH.is_file():
        raise SystemExit(f"TOML spec file not found: {_SPEC_PATH}")

    problem = load_problem_spec_toml(_SPEC_PATH)

    compiled = compile_kots_trajectory_problem(
        problem,
        kots_backend=_KOTS_BACKEND,
        jacobian_method=args.jacobian_method,
        jacobian_options=({"eps": args.numerical_eps} if args.jacobian_method == "numerical" else
                          {"mode": args.autodiff_mode, "jit": args.autodiff_jit}
                          if args.jacobian_method == "autodiff" else None),
        jacobian_strategy=args.jacobian_strategy,
        batch_trajectory=args.batch_trajectory,
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
        solver=args.solver,
        options={"max_iters": 500,
                 **({} if args.solver in ("lm-ls", "gauss_newton_krylov") else
                    {"tol_dx": 1e-8 if args.solver == "gauss_newton" else 1e-12}),
                 "history_path": args.history, "history_vectors": args.history_vectors,
                 "verbose": not args.quiet,
                 **({"line_search_history_path": args.line_search_history} if not trust_solver
                    else {"trial_history_path": args.trial_history})},
    )
    reduction.runtime.update_state_if_needed()
    out = SolveOutcome(
        solution=runtime.pack.get().copy(),
        stats=out_reduced.stats,
        timing=out_reduced.timing,
        history=out_reduced.history,
        line_search_history=out_reduced.line_search_history,
        trial_history=out_reduced.trial_history,
        meta={
            **out_reduced.meta,
            "x0": x0,
            "reduced_solution": out_reduced.solution.copy(),
        },
    )
    steps = int(compiled.trajectory_map.steps)
    if args.show_line_search:
        print(format_solver_history(out.line_search_history))
    if args.show_trials:
        print(format_solver_history(out.trial_history))

    print("=== robokots_trajectory_dynamics ===")
    print(f"derivatives={compiled.derivative_settings}")
    print(f"spec={_SPEC_PATH}")
    print(f"model={problem['model']['file']} (order={compiled.model_order}, kots_backend={_KOTS_BACKEND})")
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
                "kots_backend": _KOTS_BACKEND,
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
