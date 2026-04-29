from __future__ import annotations

from pathlib import Path

from rei import compile_nls_problem_spec_json, format_solve_report, solve

try:
    from robokots.kots import Kots
    from rei.backends.state.robotics.kots import KotsStateBuilder
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires RoboKots.\n"
        "Install dependencies (e.g. `uv sync --group kots`) and re-run:\n"
        "  PYTHONPATH=. python examples/robokots_ik.py"
    ) from e

_EXAMPLES_DIR = Path(__file__).resolve().parent
_SPEC_PATH = _EXAMPLES_DIR / "spec" / "ik_pos.json"
_MODEL_PATH = _EXAMPLES_DIR / "models" / "planar2.json"
_ORDER = 3


def main() -> None:
    if not _SPEC_PATH.is_file():
        raise SystemExit(f"JSON spec file not found: {_SPEC_PATH}")
    if not _MODEL_PATH.is_file():
        raise SystemExit(f"Model file not found: {_MODEL_PATH}")

    kots = Kots.from_json_file(str(_MODEL_PATH), order=_ORDER)
    data = kots.state_dict_

    builder = KotsStateBuilder(
        kots,
        data,
        q_var="q",
        fields=("pos",),
        dynamics_fields=None,
    )
    runtime = compile_nls_problem_spec_json(_SPEC_PATH, build_state=builder.build_state)

    x0 = runtime.pack.get().copy()
    out = solve(
        runtime,
        solver="gauss_newton",
        options={"max_iters": 200, "tol_dx": 1e-8},
    )
    x_star = out.solution
    stats = out.stats

    print("=== robokots_ik ===")
    print(f"spec={_SPEC_PATH}")
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
