from __future__ import annotations

from typing import Any

import numpy as np

Array = np.ndarray


def infer_robot_model_dof(model: Any) -> int:
    dof_fn = getattr(model, "dof", None)
    if callable(dof_fn):
        dof = int(dof_fn())
        return dof

    robot = getattr(model, "robot_", None)
    if robot is not None and hasattr(robot, "dof"):
        return int(getattr(robot, "dof"))

    raise ValueError("robotics.motion: unable to resolve model dof.")


def infer_robot_model_order(model: Any) -> int:
    order_fn = getattr(model, "order", None)
    if callable(order_fn):
        order = int(order_fn())
    else:
        order = int(getattr(model, "order_", 1))
    if order < 1:
        raise ValueError(f"robotics.motion: model order must be >= 1, got {order}.")
    return int(order)


def dof_sorted_robot_joints(model: Any) -> list[Any] | None:
    robot = getattr(model, "robot_", None)
    if robot is None:
        return None
    joints_raw = getattr(robot, "joints", None)
    if joints_raw is None:
        return None
    joints = [joint for joint in joints_raw if int(getattr(joint, "dof", 0)) > 0]
    joints.sort(key=lambda joint: int(getattr(joint, "dof_index", 0)))
    return joints


def expand_coordinate_motion_by_robot_layout(
    model: Any,
    q: Array,
    *,
    dof: int,
    order: int,
    error_prefix: str,
) -> Array:
    q_vec = np.asarray(q, dtype=float).reshape(-1)
    motion = np.zeros(int(dof) * int(order), dtype=float)
    robot = getattr(model, "robot_", None)
    if robot is None:
        for i in range(min(q_vec.size, int(dof))):
            motion[i * int(order)] = float(q_vec[i])
        return motion

    owners = [*getattr(robot, "links", []), *getattr(robot, "joints", [])]
    owners = [owner for owner in owners if int(getattr(owner, "dof", 0)) > 0]
    owners.sort(key=lambda owner: int(getattr(owner, "dof_index", 0)))

    cursor = 0
    for owner in owners:
        owner_dof = int(getattr(owner, "dof", 0))
        dof_index = int(getattr(owner, "dof_index", 0))
        start = dof_index * int(order)
        stop = start + owner_dof
        if stop > motion.size:
            raise ValueError(f"{error_prefix}: invalid dof_index/dof in robot structure.")
        motion[start:stop] = q_vec[cursor : cursor + owner_dof]
        cursor += owner_dof

    if cursor != q_vec.size:
        raise ValueError(
            f"{error_prefix}: failed to map q into motion coordinates. "
            f"Mapped {cursor} elements from q size {q_vec.size}."
        )
    return motion


def interleaved_motion_jacobian_used_order(
    *,
    dof: int,
    order: int,
    used_order: int,
    dmotiondp: Array,
) -> Array | None:
    dof_i = int(dof)
    order_i = int(order)
    used_order_i = int(used_order)
    D = np.asarray(dmotiondp, dtype=float)
    if dof_i <= 0 or used_order_i < 1 or used_order_i > order_i:
        return None
    if D.shape[0] != dof_i * order_i:
        return None

    cols = int(dof_i * used_order_i)
    out = np.zeros((cols, D.shape[1]), dtype=float)
    for i in range(dof_i):
        src0 = i * order_i
        src1 = src0 + used_order_i
        dst0 = i * used_order_i
        dst1 = dst0 + used_order_i
        out[dst0:dst1, :] = D[src0:src1, :]
    return out


__all__ = [
    "dof_sorted_robot_joints",
    "expand_coordinate_motion_by_robot_layout",
    "infer_robot_model_dof",
    "infer_robot_model_order",
    "interleaved_motion_jacobian_used_order",
]
