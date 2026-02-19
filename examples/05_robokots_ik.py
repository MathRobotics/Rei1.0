from __future__ import annotations

from pathlib import Path

from eiopt.optimize.builder import compile_nls_problem, load_problem_toml
from eiopt.optimize.report import format_solve_report
from eiopt.optimize.solvers import solve

try:
    from robokots.kots import Kots
    from eiopt.backends.state.robotics.kots import KotsStateBuilder
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
    x_star, initial_cost, cost, iters, rnorm, dxnorm, converged = solve(
        runtime,
        solver="gauss_newton",
        options={"max_iters": 200, "tol_dx": 1e-8},
    )

    print("=== 05_robokots_ik ===")
    print(f"dsl={_DSL_PATH}")
    print(f"model={_MODEL_PATH} (order={_ORDER})")
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
