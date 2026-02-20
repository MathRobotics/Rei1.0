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
    out = solve(runtime, solver="gauss_newton")
    x_star = out.solution
    stats = out.stats

    print("=== 03_toml_problem ===")
    print(f"dsl={dsl_path}")
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
