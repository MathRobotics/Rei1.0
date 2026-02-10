from __future__ import annotations

from pathlib import Path

try:
    import pinocchio as pin  # robotics library
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires the robotics `pinocchio` Python bindings.\n"
        "Install Pinocchio in your environment (e.g. via conda-forge) and re-run:\n"
        "  PYTHONPATH=. python examples/main_pinocchio.py"
    ) from e

from eiopt import compile_problem, format_solve_report, load_problem_toml, solve_gauss_newton
from eiopt.backends.pinocchio import PinocchioFramePosStateBuilder

_EXAMPLES_DIR = Path(__file__).resolve().parent
_URDF_PATH = _EXAMPLES_DIR / "models" / "planar2.urdf"
_DSL_PATH = _EXAMPLES_DIR / "dls" / "pinocchio_ik_pos.toml"


def main() -> int:
    # For a CLI version with arguments, see: examples/cli/main_pinocchio.py
    model = pin.buildModelFromUrdf(str(_URDF_PATH))
    data = model.createData()

    dsl = load_problem_toml(_DSL_PATH)

    builder = PinocchioFramePosStateBuilder(model, data, q_var="q")
    problem, ctx, required = compile_problem(dsl, build_state=builder.build_state)

    x0 = ctx.pack.get().copy()
    x_star, _cost, _iters, _rnorm, _dxnorm, _converged = solve_gauss_newton(
        problem, ctx.pack, max_iters=20, ctx=ctx, required=required
    )

    print(format_solve_report(problem, ctx=ctx, required=required, x0=x0, x_star=x_star))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
