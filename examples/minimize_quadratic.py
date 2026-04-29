from __future__ import annotations

import argparse

from rei import compile_nls_problem_spec, format_solve_report, solve


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal quadratic NLS example.")
    parser.add_argument(
        "--target",
        nargs=2,
        type=float,
        metavar=("T0", "T1"),
        default=[3.0, -1.0],
        help="Target vector for q (default: 3.0 -1.0).",
    )
    args = parser.parse_args()
    target = [float(args.target[0]), float(args.target[1])]

    spec = {
        "variables": {
            "q": {"dim": 2, "init": [0.0, 0.0]},
        },
        "terms": [
            {
                "name": "q_minus_target",
                "residual": {
                    "var": "q",
                    "target": target,
                },
            }
        ],
    }
    runtime = compile_nls_problem_spec(
        spec,
        build_state=lambda *_args, **_kwargs: {},
    )

    x0 = runtime.pack.get().copy()
    out = solve(
        runtime,
        solver="gauss_newton",
        options={"max_iters": 50, "damping": 0.0, "line_search": False},
    )
    x_star = out.solution
    stats = out.stats

    print("=== minimize_quadratic ===")
    print(f"target={target}")
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
