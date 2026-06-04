from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np

from rei.backends.state.robotics.kots import KotsTrajectoryStateBuilder
from rei.core.state_cache import OwnerKey, StateKey
from rei.core.state_schema import DTYPE_DYNAMICS
from rei.core.trajectory import TrajectoryMap

try:
    from robokots.core.state import StateType
    from robokots.kots import Kots
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This benchmark requires RoboKots. For a local checkout, run for example:\n"
        "  PYTHONPATH=/path/to/RoboKots:. python developer/benchmarks/robokots_jacobian_mul.py"
    ) from e


def _add_revolute_chain(
    links: list[dict],
    joints: list[dict],
    *,
    parent_link_id: int,
    count: int,
) -> None:
    axes = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    parent = int(parent_link_id)
    next_link_id = max(int(link["id"]) for link in links) + 1
    next_joint_id = max(int(joint["id"]) for joint in joints) + 1
    for i in range(int(count)):
        link_id = next_link_id + i
        joint_id = next_joint_id + i
        links.append(
            {
                "id": link_id,
                "name": f"chain_link{i + 1}",
                "mass": 1.0,
                "cog": [0.1, 0.0, 0.0],
                "inertia": [0.02, 0.02, 0.02, 0.0, 0.0, 0.0],
                "geometry": "generated_chain.stl",
            }
        )
        joints.append(
            {
                "id": joint_id,
                "name": f"chain_joint{i + 1}",
                "type": "revolute",
                "parent_link_id": parent,
                "child_link_id": link_id,
                "axis": list(axes[i % len(axes)]),
                "origin": {
                    "position": [0.2, 0.0, 0.05],
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                },
            }
        )
        parent = link_id


def build_serial_chain_model(dof: int) -> dict:
    links = [
        {"id": 0, "name": "world"},
        {
            "id": 1,
            "name": "base",
            "mass": 10.0,
            "cog": [0.0, 0.0, 0.0],
            "inertia": [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
            "geometry": "generated_base.stl",
        },
    ]
    joints = [
        {
            "id": 0,
            "name": "root",
            "type": "fix",
            "parent_link_id": 0,
            "child_link_id": 1,
            "origin": {
                "position": [0.0, 0.0, 0.0],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            },
        }
    ]
    _add_revolute_chain(links, joints, parent_link_id=1, count=dof)
    return {"links": links, "joints": joints}


def make_kots(dof: int, order: int) -> Kots:
    model_data = build_serial_chain_model(dof)
    try:
        return Kots.from_json_data(model_data, order=order, dim=3)
    except TypeError:
        return Kots.from_json_data(model_data, order=order)


def joint_names(kots: Kots) -> list[str]:
    return [joint.name for joint in kots.robot_.joints if int(getattr(joint, "dof", 0)) > 0]


def measure(fn: Callable[[], object], *, repeat: int, warmup: int) -> dict[str, float]:
    for _ in range(int(warmup)):
        fn()
    values = []
    for _ in range(int(repeat)):
        t0 = time.perf_counter()
        fn()
        values.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(values, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def fmt(stats: dict[str, float]) -> str:
    return (
        f"mean={stats['mean_ms']:.3f} ms "
        f"std={stats['std_ms']:.3f} "
        f"min={stats['min_ms']:.3f} "
        f"max={stats['max_ms']:.3f}"
    )


def make_trajectory_maps(
    *,
    dof: int,
    order: int,
    p_dim: int,
    rng: np.random.Generator,
) -> tuple[TrajectoryMap, dict[int, TrajectoryMap]]:
    maps = []
    for _ in range(int(order)):
        A = rng.standard_normal((dof, p_dim)) * 0.03
        maps.append(TrajectoryMap(A=A, b=np.zeros((dof,), dtype=float), steps=1, q_dim=dof))
    return maps[0], {i: maps[i] for i in range(1, int(order))}


def raw_benchmark(args: argparse.Namespace, *, dof: int) -> None:
    rng = np.random.default_rng(int(args.seed) + dof)
    kots = make_kots(dof, int(args.order))
    kots.import_motions(rng.standard_normal(dof * int(args.order)) * 0.05)
    kots.kinematics()
    kots.dynamics()

    refs = [StateType("joint", name, args.field) for name in joint_names(kots)]
    J = np.asarray(kots.jacobian(refs), dtype=float)
    rhs = rng.standard_normal((J.shape[1], int(args.rhs_cols)))
    transpose_rhs = rng.standard_normal((J.shape[0], int(args.rhs_cols)))

    diff_mul = float(np.max(np.abs((J @ rhs) - np.asarray(kots.jacobian_mul(refs, rhs), dtype=float))))
    diff_transpose = float(
        np.max(np.abs((J.T @ transpose_rhs) - np.asarray(kots.jacobian_transpose_mul(refs, transpose_rhs), dtype=float)))
    )

    s_jp = measure(lambda: kots.jacobian(refs) @ rhs, repeat=int(args.repeat), warmup=int(args.warmup))
    s_mul = measure(lambda: kots.jacobian_mul(refs, rhs), repeat=int(args.repeat), warmup=int(args.warmup))
    s_jtr = measure(lambda: kots.jacobian(refs).T @ transpose_rhs, repeat=int(args.repeat), warmup=int(args.warmup))
    s_tr = measure(lambda: kots.jacobian_transpose_mul(refs, transpose_rhs), repeat=int(args.repeat), warmup=int(args.warmup))

    print(f"\nraw RoboKots dof={dof} J={J.shape} field={args.field}")
    print(f"  jacobian @ RHS          {fmt(s_jp)}")
    print(f"  jacobian_mul            {fmt(s_mul)} ratio={s_mul['mean_ms'] / s_jp['mean_ms']:.3f}x diff={diff_mul:.3e}")
    print(f"  jacobian.T @ RHS        {fmt(s_jtr)}")
    print(f"  jacobian_transpose_mul  {fmt(s_tr)} ratio={s_tr['mean_ms'] / s_jtr['mean_ms']:.3f}x diff={diff_transpose:.3e}")


def rei_benchmark(args: argparse.Namespace, *, dof: int) -> None:
    rng = np.random.default_rng(int(args.seed) + 1000 + dof)
    p_dim = int(args.p_dim) if args.p_dim is not None else dof * int(args.p_dim_factor)
    trajectory_map, derivative_maps = make_trajectory_maps(
        dof=dof,
        order=int(args.order),
        p_dim=p_dim,
        rng=rng,
    )
    p = rng.standard_normal(p_dim) * 0.1
    key = StateKey(
        k=0,
        owner=OwnerKey("total_joint", "total_joint"),
        dtype=DTYPE_DYNAMICS,
        field=f"{args.rei_field}_J_p",
    )

    values = {}
    timings = {}
    for strategy in ("dense", "mul"):
        kots = make_kots(dof, int(args.order))
        builder = KotsTrajectoryStateBuilder(
            kots,
            kots.state_dict_,
            trajectory_map=trajectory_map,
            trajectory_derivative_maps=derivative_maps,
            dynamics_fields=[args.rei_field],
            jacobian_strategy=strategy,
        )

        def op() -> np.ndarray:
            return np.asarray(builder.build_state(p, required=[key])[key], dtype=float)

        values[strategy] = op()
        timings[strategy] = measure(op, repeat=int(args.repeat), warmup=int(args.warmup))

    rhs = rng.standard_normal((values["dense"].shape[0], int(args.rhs_cols)))
    dense_kots = make_kots(dof, int(args.order))
    dense_builder = KotsTrajectoryStateBuilder(
        dense_kots,
        dense_kots.state_dict_,
        trajectory_map=trajectory_map,
        trajectory_derivative_maps=derivative_maps,
        dynamics_fields=[args.rei_field],
        jacobian_strategy="dense",
    )
    fast_kots = make_kots(dof, int(args.order))
    fast_builder = KotsTrajectoryStateBuilder(
        fast_kots,
        fast_kots.state_dict_,
        trajectory_map=trajectory_map,
        trajectory_derivative_maps=derivative_maps,
        dynamics_fields=[args.rei_field],
        jacobian_strategy="mul",
    )

    def dense_transpose_op() -> np.ndarray:
        Jp = np.asarray(dense_builder.build_state(p, required=[key])[key], dtype=float)
        return Jp.T @ rhs

    def transpose_op() -> np.ndarray:
        value_key = StateKey(k=0, owner=key.owner, dtype=key.dtype, field=args.rei_field)
        return np.asarray(fast_builder.param_jacobian_transpose_mul(p, value_key, rhs), dtype=float)

    dense_t = dense_transpose_op()
    fast_t = transpose_op()
    s_dense_t = measure(dense_transpose_op, repeat=int(args.repeat), warmup=int(args.warmup))
    s_fast_t = measure(transpose_op, repeat=int(args.repeat), warmup=int(args.warmup))

    diff_mul = float(np.max(np.abs(values["dense"] - values["mul"])))
    diff_transpose = float(np.max(np.abs(dense_t - fast_t)))

    print(f"\nRei build_state dof={dof} Jp={values['dense'].shape} p_dim={p_dim} field={args.rei_field}")
    print(f"  dense                  {fmt(timings['dense'])}")
    print(f"  mul                    {fmt(timings['mul'])} ratio={timings['mul']['mean_ms'] / timings['dense']['mean_ms']:.3f}x diff={diff_mul:.3e}")
    print(f"  dense Jp.T @ RHS       {fmt(s_dense_t)}")
    print(f"  transpose fast path    {fmt(s_fast_t)} ratio={s_fast_t['mean_ms'] / s_dense_t['mean_ms']:.3f}x diff={diff_transpose:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RoboKots dense Jacobian and multiply APIs through raw and Rei paths.")
    parser.add_argument("--dofs", nargs="+", type=int, default=[7, 32])
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--field", default="torque_diff2", help="Raw RoboKots StateType field.")
    parser.add_argument("--rei-field", default="torque_d2", help="Canonical Rei dynamics field.")
    parser.add_argument("--rhs-cols", type=int, default=350)
    parser.add_argument("--p-dim", type=int, default=None)
    parser.add_argument("--p-dim-factor", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    for dof in args.dofs:
        raw_benchmark(args, dof=int(dof))
        rei_benchmark(args, dof=int(dof))


if __name__ == "__main__":
    main()
