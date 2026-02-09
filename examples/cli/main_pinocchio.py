from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import pinocchio as pin  # robotics library
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires the robotics `pinocchio` Python bindings.\n"
        "Install Pinocchio in your environment (e.g. via conda-forge) and re-run:\n"
        "  PYTHONPATH=. python examples/cli/main_pinocchio.py"
    ) from e

from eiopt import compile_problem, format_solve_report, load_problem_toml, solve_gauss_newton
from eiopt.backends.pinocchio import PinocchioFramePosStateBuilder
from eiopt.dsl.dsl_ops import find_const_expr, find_var_spec, rewrite_get_state_owner_name

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_URDF_PATH = _EXAMPLES_DIR / "models" / "planar2.urdf"
_DEFAULT_SPEC_PATH = _EXAMPLES_DIR / "dls" / "pinocchio_ik_pos.toml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pinocchio IK example for EiOpt.")
    parser.add_argument("--urdf", type=Path, default=_DEFAULT_URDF_PATH, help="Path to URDF.")
    parser.add_argument("--spec", type=Path, default=_DEFAULT_SPEC_PATH, help="Path to problem spec TOML.")
    parser.add_argument("--ee", type=str, default="ee", help="End-effector frame name.")
    parser.add_argument("--report", action="store_true", help="Print a concise expr/term report after solving.")
    args = parser.parse_args(argv)

    model = pin.buildModelFromUrdf(str(args.urdf))
    data = model.createData()

    dsl = load_problem_toml(args.spec)

    ee_frame = str(args.ee)
    rewrite_get_state_owner_name(dsl, dtype="frame", owner_type="link", owner_name=ee_frame)

    q_var_spec = find_var_spec(dsl, name="q")
    if q_var_spec is None:
        raise SystemExit("Spec must declare a variable named 'q'.")
    q0 = np.asarray(q_var_spec["init"], dtype=float).reshape(-1)

    state_builder = PinocchioFramePosStateBuilder(model, data, q_var="q")
    problem, ctx, required = compile_problem(dsl, build_state=state_builder.build_state, model=model)

    ctx.state.update_if_needed(ctx.pack, time=ctx.time, required=required)
    key_pos = next(
        k
        for k in required
        if getattr(k, "dtype", None) == "frame"
        and getattr(k, "field", None) == "pos"
        and getattr(getattr(k, "owner", None), "owner_name", None) == ee_frame
    )
    ee_pos0 = np.asarray(ctx.state.get(key_pos), dtype=float).reshape(3)

    target_expr = find_const_expr(dsl, name="target_pos")
    if target_expr is None:
        raise SystemExit("DSL must contain a const expr with name='target_pos'.")
    target = np.asarray(target_expr.get("value", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)

    print("ee_pos0:", ee_pos0)
    print("target:", target)

    x0 = ctx.pack.get().copy()
    x_star, _cost, _iters, _rnorm, _dxnorm, _converged = solve_gauss_newton(
        problem, ctx.pack, max_iters=20, ctx=ctx, required=required
    )

    ctx.state.update_if_needed(ctx.pack, time=ctx.time, required=required)
    ee_pos_star = np.asarray(ctx.state.get(key_pos), dtype=float).reshape(3)

    print("q0:", q0)
    print("q*:", x_star)
    print("ee_pos*:", ee_pos_star)
    if args.report:
        print(format_solve_report(problem, ctx=ctx, required=required, x0=x0, x_star=x_star))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
