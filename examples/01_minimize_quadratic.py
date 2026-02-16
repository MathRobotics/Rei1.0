from __future__ import annotations

import argparse

from eiopt.optimize.builder import compile_nls_problem
from eiopt.optimize.report import format_solve_report
from eiopt.optimize.solvers import solve


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

    dsl = {
        "variables": [
            {"name": "q", "dim": 2, "init": [0.0, 0.0]},
        ],
        "terms": [
            {
                "expr": {
                    "type": "sub",
                    "name": "q_minus_target",
                    "a": {"type": "get_var", "var": "q"},
                    "b": {"type": "const", "var": "q", "value": target},
                },
                "cost": {"type": "l2"},
            }
        ],
    }

    runtime = compile_nls_problem(
        dsl,
        build_state=lambda *_args, **_kwargs: {},
    )

    x0 = runtime.pack.get().copy()
    x_star, cost, iters, rnorm, dxnorm, converged = solve(
        runtime,
        solver="gauss_newton",
        max_iters=50,
        gn_damping=0.0,
        gn_line_search=False,
    )

    print("=== 01_minimize_quadratic ===")
    print(f"target={target}")
    print(f"converged={converged} iters={iters} cost={cost:.3e} rnorm={rnorm:.3e} dxnorm={dxnorm:.3e}")
    print(f"x0={x0}")
    print(f"x*={x_star}")
    print(format_solve_report(runtime, x0=x0, x_star=x_star))


if __name__ == "__main__":
    main()
