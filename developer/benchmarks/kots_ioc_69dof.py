"""Measure the representative RoboKots-Rust IOC VJP workload.

Run with ``uv run --group kots python developer/benchmarks/kots_ioc_69dof.py``.
This is intentionally a reporting benchmark, not a wall-clock CI assertion;
CI uses the structural batching guard in ``tests/test_kots_builder_dynamics_mock.py``.
"""
from __future__ import annotations

import argparse
import statistics
from time import perf_counter
from typing import Any

import numpy as np

from rei.optimize_backends.trajectory_ioc import (
    compile_trajectory_ioc_problem,
    estimate_ioc_weights,
)

# Running this file directly places its directory on ``sys.path``.
from robokots_jacobian_mul import build_serial_chain_model

try:
    from robokots.kots import Kots
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install the kots dependency group to run this benchmark.") from exc


def _make_kots(*, dof: int, order: int) -> Any:
    data = build_serial_chain_model(dof)
    try:
        return Kots.from_json_data(data, order=order, dim=3)
    except TypeError:
        return Kots.from_json_data(data, order=order)


def _state_stack(*, name: str, field: str, steps: int) -> dict[str, Any]:
    return {
        "type": "vstack",
        "name": name,
        "parts": [
            {
                "type": "get_state",
                "name": f"{name}_{k}",
                "key": {
                    "k": k,
                    "owner_type": "total_joint",
                    "owner_name": "robot",
                    "dtype": "dynamics",
                    "field": field,
                },
                "jac": {"var": "p"},
            }
            for k in range(steps)
        ],
    }


def make_dsl(*, frames: int, dof: int, controls: int) -> dict[str, Any]:
    derivatives = [
        {"type": "get_traj_var", "name": "q", "var": "p"},
        {"type": "get_traj_var", "name": "qdot", "var": "p", "derivative_order": 1},
        {"type": "get_traj_var", "name": "qddot", "var": "p", "derivative_order": 2},
        {"type": "get_traj_var", "name": "qdddot", "var": "p", "derivative_order": 3},
        _state_stack(name="torque", field="torque", steps=frames),
        _state_stack(name="torque_d1", field="torque_d1", steps=frames),
    ]
    return {
        "time": {"N": frames - 1, "dt": 0.02},
        "trajectory": {
            "type": "bspline", "var": "p", "q_dim": dof,
            "degree": 3, "num_ctrl_points": controls,
        },
        "variables": [{"name": "p", "init": {"fill": 0.01}}],
        "terms": [{"expr": expr, "cost": {"type": "l2"}} for expr in derivatives],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--dof", type=int, default=69)
    parser.add_argument("--controls", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.frames < 2 or args.dof < 1 or args.controls < 4:
        raise SystemExit("frames >= 2, dof >= 1, and controls >= 4 are required.")

    compiled = compile_trajectory_ioc_problem(
        make_dsl(frames=args.frames, dof=args.dof, controls=args.controls),
        backend="kots", model=_make_kots(dof=args.dof, order=4), data=None,
        kots_backend="rust", dynamics_fields=("torque", "torque_d1"),
    )
    for _ in range(args.warmup):
        estimate_ioc_weights(compiled)
    samples: list[float] = []
    for _ in range(args.repeat):
        t0 = perf_counter()
        result = estimate_ioc_weights(compiled)
        samples.append(1000.0 * (perf_counter() - t0))
    print(
        f"estimate_ioc_weights: frames={args.frames} dof={args.dof} "
        f"controls={args.controls} backend=rust "
        f"median={statistics.median(samples):.3f} ms "
        f"min={min(samples):.3f} ms max={max(samples):.3f} ms "
        f"terms={len(result['terms'])}"
    )


if __name__ == "__main__":
    main()
