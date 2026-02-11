from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..core.trajectory import TrajectoryMap


def pick_trajectory_value(dsl: Mapping[str, Any], *, section: str, key: str) -> Any:
    if key in dsl:
        return dsl[key]
    section_obj = dsl.get(section, None)
    if isinstance(section_obj, Mapping) and key in section_obj:
        return section_obj[key]
    return None


def resolve_optional_positive_int(value: Any, *, name: str, fallback: int | None = None) -> int | None:
    v = fallback if value is None else value
    if v is None:
        return None
    try:
        out = int(v)
    except Exception as e:
        raise ValueError(f"trajectory.{name} must be an integer, got {v!r}.") from e
    if out <= 0:
        raise ValueError(f"trajectory.{name} must be > 0, got {out}.")
    return out


def resolve_required_positive_int(value: Any, *, name: str) -> int:
    if value is None:
        raise ValueError(f"trajectory.{name} is required.")
    try:
        out = int(value)
    except Exception as e:
        raise ValueError(f"trajectory.{name} must be an integer, got {value!r}.") from e
    if out <= 0:
        raise ValueError(f"trajectory.{name} must be > 0, got {out}.")
    return out


def resolve_required_nonnegative_int(value: Any, *, name: str) -> int:
    if value is None:
        raise ValueError(f"trajectory.{name} is required.")
    try:
        out = int(value)
    except Exception as e:
        raise ValueError(f"trajectory.{name} must be an integer, got {value!r}.") from e
    if out < 0:
        raise ValueError(f"trajectory.{name} must be >= 0, got {out}.")
    return out


def default_steps_from_time(time: Any) -> int | None:
    if time is None:
        return None
    if hasattr(time, "N"):
        try:
            return int(time.N) + 1
        except Exception:
            return None
    if isinstance(time, Mapping):
        time_dsl = time.get("time", time)
        if isinstance(time_dsl, Mapping) and "N" in time_dsl:
            try:
                return int(time_dsl["N"]) + 1
            except Exception:
                return None
    return None


def infer_bspline_q_dim_from_var(traj_dsl: Mapping[str, Any], *, var_dim: int) -> int | None:
    bspline = traj_dsl.get("bspline", None)
    n_ctrl_raw = bspline.get("num_ctrl_points", None) if isinstance(bspline, Mapping) else None
    if n_ctrl_raw is None:
        n_ctrl_raw = traj_dsl.get("num_ctrl_points", None)
    if n_ctrl_raw is None:
        return None
    try:
        n_ctrl = int(n_ctrl_raw)
    except Exception:
        return None
    if n_ctrl <= 0 or int(var_dim) <= 0 or int(var_dim) % n_ctrl != 0:
        return None
    q_dim = int(var_dim) // n_ctrl
    return q_dim if q_dim > 0 else None


def build_trajectory_map(
    traj_dsl: Mapping[str, Any],
    *,
    default_steps: int | None = None,
    default_q_dim: int | None = None,
) -> TrajectoryMap:
    """Build a ``TrajectoryMap`` from DSL ``[trajectory]`` config."""

    if not isinstance(traj_dsl, Mapping):
        raise TypeError("build_trajectory_map: trajectory dsl must be a mapping.")

    typ = str(traj_dsl.get("type", "")).strip().lower()
    if typ == "":
        raise ValueError("build_trajectory_map: trajectory.type is required.")

    steps = resolve_optional_positive_int(
        pick_trajectory_value(traj_dsl, section=typ, key="steps"),
        name="steps",
        fallback=default_steps,
    )
    q_dim = resolve_optional_positive_int(
        pick_trajectory_value(traj_dsl, section=typ, key="q_dim"),
        name="q_dim",
        fallback=default_q_dim,
    )

    if typ == "bspline":
        if steps is None:
            raise ValueError(
                "build_trajectory_map: steps is required for bspline trajectory "
                "(set trajectory.steps or pass default_steps)."
            )
        if q_dim is None:
            raise ValueError(
                "build_trajectory_map: q_dim is required for bspline trajectory "
                "(set trajectory.q_dim or pass default_q_dim)."
            )

        degree = resolve_required_nonnegative_int(
            pick_trajectory_value(traj_dsl, section="bspline", key="degree"),
            name="degree",
        )
        num_ctrl_points = resolve_required_positive_int(
            pick_trajectory_value(traj_dsl, section="bspline", key="num_ctrl_points"),
            name="num_ctrl_points",
        )
        knot_vector_raw = pick_trajectory_value(traj_dsl, section="bspline", key="knot_vector")
        u_samples_raw = pick_trajectory_value(traj_dsl, section="bspline", key="u_samples")
        knot_vector = None if knot_vector_raw is None else np.asarray(knot_vector_raw, dtype=float).reshape(-1)
        u_samples = None if u_samples_raw is None else np.asarray(u_samples_raw, dtype=float).reshape(-1)
        return TrajectoryMap.from_bspline(
            steps=steps,
            q_dim=q_dim,
            degree=degree,
            num_ctrl_points=num_ctrl_points,
            knot_vector=knot_vector,
            u_samples=u_samples,
        )

    if typ == "linear":
        a_raw = pick_trajectory_value(traj_dsl, section="linear", key="A")
        if a_raw is None:
            raise ValueError("build_trajectory_map: trajectory.linear.A is required for type='linear'.")
        try:
            a_arr = np.asarray(a_raw, dtype=float)
        except Exception as e:
            raise ValueError("build_trajectory_map: failed to parse linear A as numeric array.") from e

        if a_arr.ndim == 1:
            if steps is None or q_dim is None:
                raise ValueError(
                    "build_trajectory_map: steps and q_dim are required when linear A is 1D "
                    "(flattened array)."
                )
            rows = int(steps * q_dim)
            if rows <= 0:
                raise ValueError("build_trajectory_map: invalid steps*q_dim for linear A reshape.")
            if a_arr.size % rows != 0:
                raise ValueError(
                    "build_trajectory_map: linear A size mismatch. "
                    f"Expected multiple of {rows} (=steps*q_dim), got {a_arr.size}."
                )
            a_mat = a_arr.reshape(rows, -1)
        elif a_arr.ndim == 2:
            a_mat = a_arr
        else:
            raise ValueError(
                "build_trajectory_map: linear A must be 1D(flat) or 2D(matrix), "
                f"got ndim={a_arr.ndim}."
            )

        rows = int(a_mat.shape[0])
        if steps is None and q_dim is None:
            raise ValueError(
                "build_trajectory_map: cannot infer both steps and q_dim from linear A only. "
                "Provide trajectory.steps or trajectory.q_dim (or defaults)."
            )
        if steps is None:
            if q_dim is None or rows % q_dim != 0:
                raise ValueError(
                    "build_trajectory_map: failed to infer steps from linear A rows and q_dim. "
                    f"rows={rows}, q_dim={q_dim}."
                )
            steps = int(rows // q_dim)
        if q_dim is None:
            if steps <= 0 or rows % steps != 0:
                raise ValueError(
                    "build_trajectory_map: failed to infer q_dim from linear A rows and steps. "
                    f"rows={rows}, steps={steps}."
                )
            q_dim = int(rows // steps)
        if int(steps * q_dim) != rows:
            raise ValueError(
                "build_trajectory_map: linear A row mismatch against steps and q_dim. "
                f"rows={rows}, steps*q_dim={steps * q_dim}."
            )

        b_raw = pick_trajectory_value(traj_dsl, section="linear", key="b")
        if b_raw is None:
            b_vec = np.zeros((rows,), dtype=float)
        else:
            b_vec = np.asarray(b_raw, dtype=float).reshape(-1)
            if b_vec.size != rows:
                raise ValueError(
                    "build_trajectory_map: linear b size mismatch. "
                    f"Expected {rows}, got {b_vec.size}."
                )
        return TrajectoryMap(A=a_mat, b=b_vec, steps=steps, q_dim=q_dim)

    raise ValueError(
        f"build_trajectory_map: unsupported trajectory type {typ!r}. "
        "Supported types: 'bspline', 'linear'."
    )
