from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pinocchio as pin  # robotics library
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires the `pinocchio` Python module.\n"
        "Install Pinocchio in your environment (e.g. via conda-forge), then re-run:\n"
        "  python main_pinocchio.py"
    ) from e

from eiopt import compile_problem, load_problem_toml, solve_gauss_newton, with_standard_joint_q
from eiopt.backends.pinocchio import PinocchioFramePosStateBuilder
from eiopt.dsl.dsl_ops import find_const_expr, find_var_spec, rewrite_get_state_owner_name

URDF_PATH = "models" / "planar2.urdf"
DSL_PATH = "specs" / "pinocchio_ik_pos.toml"


def main(a) -> int:
    

    model = pin.buildModelFromUrdf(str(Path(URDF_PATH)))
    data = model.createData()

    dsl_path = Path(DSL_PATH)
    dsl = load_problem_toml(dsl_path)

    ee_frame = str("ee")
    rewrite_get_state_owner_name(dsl, dtype="frame", owner_type="link", owner_name=ee_frame)

    q_var_spec = find_var_spec(dsl, name="q")
    q0 = np.asarray(q_var_spec["init"], dtype=float).reshape(-1)

    state_builder = PinocchioFramePosStateBuilder(model, data, q_var="q")
    build_state = with_standard_joint_q(state_builder.build_state, q_var="q")
    problem, ctx, required = compile_problem(dsl, build_state=build_state, model=model)

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

    solve_gauss_newton(problem, ctx.pack, max_iters=20, ctx=ctx, required=required)

    ctx.state.update_if_needed(ctx.pack, time=ctx.time, required=required)
    ee_pos_star = np.asarray(ctx.state.get(key_pos), dtype=float).reshape(3)

    print("q0:", q0)
    print("q*:", ctx.pack.vars[0].x)
    print("ee_pos*:", ee_pos_star)

    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
