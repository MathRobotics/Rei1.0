from __future__ import annotations

from pathlib import Path

from rei import compile_nls_problem_spec_json, format_solve_report, solve


def main() -> None:
    spec_path = Path(__file__).resolve().parent / "spec" / "basic.json"

    runtime = compile_nls_problem_spec_json(
        spec_path,
        build_state=lambda *_args, **_kwargs: {},
    )

    x0 = runtime.pack.get().copy()
    out = solve(runtime, solver="gauss_newton")
    x_star = out.solution
    stats = out.stats

    print("=== json_spec_problem ===")
    print(f"spec={spec_path}")
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
