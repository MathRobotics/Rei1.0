from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import robokots as kots

from eiopt import build_problem_from_spec, solve_gauss_newton, with_standard_joint_q
from eiopt.backends.robokots import PinocchioFramePosStateBuilder
from eiopt.spec.json_ops import find_const_expr, find_var_spec, rewrite_get_state_owner_name, set_const_value

URDF_PATH = "models" / "planar2.urdf"
SPEC_PATH = "specs" / "pinocchio_ik_pos.json"


def main(a) -> int:
    

    model = pin.buildModelFromUrdf(str(Path(URDF_PATH)))
    data = model.createData()

    spec_path = Path(SPEC_PATH) 
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    ee_frame = str("ee")
    rewrite_get_state_owner_name(spec, dtype="frame", owner_type="link", owner_name=ee_frame)

    q_var_spec = find_var_spec(spec, name="q")
    q0 = np.asarray(q_var_spec["init"], dtype=float).reshape(-1)

    state_builder = PinocchioFramePosStateBuilder(model, data, q_var="q")
    build_state = with_standard_joint_q(state_builder.build_state, q_var="q")
    problem, ctx, required = build_problem_from_spec(spec, build_state=build_state, model=model)

    ctx.state.update_if_needed(ctx.pack, time=ctx.time, required=required)
    key_pos = next(
        k
        for k in required
        if getattr(k, "dtype", None) == "frame"
        and getattr(k, "field", None) == "pos"
        and getattr(getattr(k, "owner", None), "owner_name", None) == ee_frame
    )
    ee_pos0 = np.asarray(ctx.state.get(key_pos), dtype=float).reshape(3)

    target_expr = find_const_expr(spec, name="target_pos")
    if target_expr is None:
        raise SystemExit("Spec must contain a const expr with name='target_pos'.")
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
