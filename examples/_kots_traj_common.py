from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from eiopt.core.state_schema import (
    DEFAULT_FRAME,
    DTYPE_DYNAMICS,
    DTYPE_KINEMATICS,
    canonical_field_name,
)
from eiopt.dsl.trajectory import build_trajectory_map_with_derivative


def collect_ee_pos_traj(runtime: Any, *, steps: int) -> np.ndarray:
    return np.asarray(
        runtime.collect_state_traj(
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
            ks=range(int(steps)),
            frame=DEFAULT_FRAME,
            expected_dim=3,
        ),
        dtype=float,
    )


def collect_joint_dynamics_traj(
    runtime: Any,
    *,
    steps: int,
    field: str = "tau",
    owner_name: str = "robot",
) -> np.ndarray:
    field_name = canonical_field_name(str(field))
    return np.asarray(
        runtime.collect_state_traj(
            owner_type="total_joint",
            owner_name=str(owner_name),
            dtype=DTYPE_DYNAMICS,
            field=field_name,
            ks=range(int(steps)),
        ),
        dtype=float,
    )


def collect_joint_torque_traj(runtime: Any, *, steps: int) -> np.ndarray:
    return collect_joint_dynamics_traj(runtime, steps=steps, field="tau")


def collect_target_waypoints(
    dsl: Mapping[str, Any],
    *,
    owner_name: str = "ee",
    dtype: str = DTYPE_KINEMATICS,
    field: str = "pos",
    target_dim: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    waypoints: list[tuple[int, np.ndarray]] = []

    for term in dsl.get("terms", []):
        expr = term.get("expr", {}) if isinstance(term, dict) else {}
        if not isinstance(expr, dict) or expr.get("type") != "sub":
            continue

        a = expr.get("a", {})
        b = expr.get("b", {})
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        if a.get("type") != "get_state" or b.get("type") != "const":
            continue

        key = a.get("key", {})
        if not isinstance(key, dict):
            continue
        if key.get("owner_name") != owner_name or key.get("dtype") != dtype or key.get("field") != field:
            continue

        k_raw = key.get("k")
        if k_raw is None:
            continue
        k = int(k_raw)

        v = np.asarray(b.get("value"), dtype=float).reshape(-1)
        if v.size != int(target_dim):
            raise ValueError(
                f"target waypoint size mismatch at k={k}. "
                f"Expected {int(target_dim)}, got {v.size}."
            )
        waypoints.append((k, v))

    if not waypoints:
        return np.zeros((0,), dtype=int), np.zeros((0, int(target_dim)), dtype=float)

    waypoints.sort(key=lambda kv: kv[0])
    target_ks = np.asarray([k for k, _v in waypoints], dtype=int)
    target_values = np.vstack([v for _k, v in waypoints])
    return target_ks, target_values


def analytic_joint_velocity(
    p_opt: np.ndarray,
    *,
    traj_dsl: Mapping[str, Any],
    steps: int,
    q_dim: int,
    dt: float,
) -> np.ndarray:
    traj_d1 = build_trajectory_map_with_derivative(
        traj_dsl,
        derivative_order=1,
        derivative_wrt="time",
        default_steps=int(steps),
        default_q_dim=int(q_dim),
        default_dt=float(dt),
    )
    p = np.asarray(p_opt, dtype=float).reshape(-1)
    if p.size != traj_d1.p_dim:
        raise ValueError(f"parameter size mismatch: expected {traj_d1.p_dim}, got {p.size}.")
    return np.asarray(traj_d1.A @ p + traj_d1.b, dtype=float).reshape(traj_d1.steps, traj_d1.q_dim)
