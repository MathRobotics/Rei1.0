from __future__ import annotations

from pathlib import Path

from rei.optimize.builder import compile_nls_problem, load_problem_toml
from rei.optimize.report import format_solve_report
from rei.optimize.solvers import solve

try:
    from robokots.kots import Kots
    from rei.backends.state.robotics.kots import KotsStateBuilder
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires RoboKots.\n"
        "Install dependencies (e.g. `uv sync --group kots`) and re-run:\n"
        "  PYTHONPATH=. python examples/05_robokots_ik.py"
    ) from e

_EXAMPLES_DIR = Path(__file__).resolve().parent
_DSL_PATH = _EXAMPLES_DIR / "dsl" / "ik_pos.toml"
_MODEL_PATH = _EXAMPLES_DIR / "models" / "planar2.json"
_ORDER = 3


def main() -> None:
    if not _DSL_PATH.is_file():
        raise SystemExit(f"DSL file not found: {_DSL_PATH}")
    if not _MODEL_PATH.is_file():
        raise SystemExit(f"Model file not found: {_MODEL_PATH}")

    kots = Kots.from_json_file(str(_MODEL_PATH), order=_ORDER)
    data = kots.state_dict_
    dsl = load_problem_toml(_DSL_PATH)

    builder = KotsStateBuilder(
        kots,
        data,
        q_var="q",
        fields=("pos",),
        dynamics_fields=None,
    )
    runtime = compile_nls_problem(dsl, build_state=builder.build_state)

    x0 = runtime.pack.get().copy()
    out = solve(
        runtime,
        solver="gauss_newton",
        options={"max_iters": 200, "tol_dx": 1e-8},
    )
    x_star = out.solution
    stats = out.stats

    print("=== 05_robokots_ik ===")
    print(f"dsl={_DSL_PATH}")
    print(f"model={_MODEL_PATH} (order={_ORDER})")
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
