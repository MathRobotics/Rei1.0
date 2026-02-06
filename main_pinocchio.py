from __future__ import annotations

import argparse
import json
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

from eiopt import build_problem_from_spec, solve_gauss_newton, with_standard_joint_q
from eiopt.backends.pinocchio import PinocchioFramePosStateBuilder
from eiopt.spec.json_ops import find_const_expr, find_var_spec, rewrite_get_state_owner_name, set_const_value

_EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
_DEFAULT_URDF_PATH = _EXAMPLES_DIR / "models" / "planar2.urdf"
_DEFAULT_SPEC_PATH = _EXAMPLES_DIR / "specs" / "pinocchio_ik_pos.json"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--urdf",
        type=str,
        default=str(_DEFAULT_URDF_PATH),
        help="Path to a URDF file (default: examples/models/planar2.urdf).",
    )
    p.add_argument(
        "--spec",
        type=str,
        default=str(_DEFAULT_SPEC_PATH),
        help="Path to an eiopt JSON spec (default: examples/specs/pinocchio_ik_pos.json).",
    )
    p.add_argument("--ee", type=str, default="ee", help="End-effector frame name (default: ee).")
    p.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Target position in world (overrides spec const name='target_pos').",
    )
    p.add_argument("--list-frames", action="store_true", help="Print model frame names and exit.")
    p.add_argument("--max-iters", type=int, default=20, help="Gauss-Newton iterations (default: 20).")
    args = p.parse_args(argv)

    urdf_path = Path(args.urdf)
    spec_path = Path(args.spec)

    if not urdf_path.is_file():
        raise SystemExit(f"URDF not found: {urdf_path}")
    if not spec_path.is_file():
        raise SystemExit(f"Spec JSON not found: {spec_path}")

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()

    if args.list_frames:
        for f in model.frames:
            print(f.name)
        return 0

    if int(getattr(model, "nq", 0)) != int(getattr(model, "nv", 0)):
        raise SystemExit(
            "This minimal example assumes an Euclidean configuration (nq == nv).\n"
            "For floating-base/quaternion models, integrate on-manifold (not shown here)."
        )

    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    ee_frame = str(args.ee)
    rewrite_get_state_owner_name(spec, dtype="frame", owner_type="link", owner_name=ee_frame)

    if args.target is not None:
        try:
            set_const_value(spec, name="target_pos", value=[float(x) for x in args.target])
        except KeyError:
            raise SystemExit("Spec must contain a const expr with name='target_pos' to use --target.") from None

    q_var_spec = find_var_spec(spec, name="q")
    if q_var_spec is None or "init" not in q_var_spec:
        raise SystemExit("Spec must contain variables: [{name:'q', init:[...]}].")
    q0 = np.asarray(q_var_spec["init"], dtype=float).reshape(-1)
    if q0.size != int(getattr(model, "nq", 0)):
        raise SystemExit(f"Spec q init has dim={q0.size}, but model.nq={int(getattr(model, 'nq', 0))}.")

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

    solve_gauss_newton(problem, ctx.pack, max_iters=int(args.max_iters), ctx=ctx, required=required)

    ctx.state.update_if_needed(ctx.pack, time=ctx.time, required=required)
    ee_pos_star = np.asarray(ctx.state.get(key_pos), dtype=float).reshape(3)

    print("q0:", q0)
    print("q*:", ctx.pack.vars[0].x)
    print("ee_pos*:", ee_pos_star)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
