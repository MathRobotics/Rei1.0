from __future__ import annotations

from pathlib import Path

from eiopt.optimize.builder import compile_nls_problem, load_problem_toml
from eiopt.optimize.report import format_solve_report
from eiopt.optimize.solvers import solve


def main() -> None:
    dsl_path = Path(__file__).resolve().parent / "dsl" / "basic.toml"
    dsl = load_problem_toml(dsl_path)

    runtime = compile_nls_problem(
        dsl,
        build_state=lambda *_args, **_kwargs: {},
    )

    x0 = runtime.pack.get().copy()
    x_star, initial_cost, cost, iters, rnorm, dxnorm, converged = solve(runtime, solver="gauss_newton")

    print("=== 03_toml_problem ===")
    print(f"dsl={dsl_path}")
    print(
        f"converged={converged} iters={iters} "
        f"cost0={initial_cost:.3e} cost={cost:.3e} "
        f"rnorm={rnorm:.3e} dxnorm={dxnorm:.3e}"
    )
    print(f"x0={x0}")
    print(f"x*={x_star}")
    print(format_solve_report(runtime, x0=x0, x_star=x_star))


if __name__ == "__main__":
    main()
