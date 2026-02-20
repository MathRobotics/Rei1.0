from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .bspline import default_clamped_uniform_knots
from .trajectory import TrajectoryMap


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


def default_dt_from_time(time: Any) -> float | None:
    if time is None:
        return None
    if hasattr(time, "dt"):
        try:
            return float(time.dt)
        except Exception:
            return None
    if isinstance(time, Mapping):
        time_dsl = time.get("time", time)
        if isinstance(time_dsl, Mapping) and "dt" in time_dsl:
            try:
                return float(time_dsl["dt"])
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


def _first_derivative_operator(*, steps: int, step_size: float) -> np.ndarray:
    """Finite-difference derivative operator with edge one-sided stencils.

    Returns matrix ``G`` such that ``dq/dx ~= G @ q`` for a single scalar trajectory.
    """

    steps = int(steps)
    h = float(step_size)
    if steps < 2:
        raise ValueError(f"_first_derivative_operator: steps must be >= 2, got {steps}.")
    if h <= 0.0:
        raise ValueError(f"_first_derivative_operator: step_size must be > 0, got {h}.")

    G = np.zeros((steps, steps), dtype=float)
    if steps == 2:
        G[0, 0] = -1.0 / h
        G[0, 1] = 1.0 / h
        G[1, 0] = -1.0 / h
        G[1, 1] = 1.0 / h
        return G

    G[0, 0] = -1.0 / h
    G[0, 1] = 1.0 / h
    G[-1, -2] = -1.0 / h
    G[-1, -1] = 1.0 / h

    c = 0.5 / h
    for i in range(1, steps - 1):
        G[i, i - 1] = -c
        G[i, i + 1] = c
    return G


def _finite_difference_maps(
    base: TrajectoryMap,
    *,
    max_derivative_order: int,
    step_size: float,
) -> list[TrajectoryMap]:
    """Approximate derivative trajectory maps via finite-difference operators."""

    max_order = int(max_derivative_order)
    if max_order < 0:
        raise ValueError(f"_finite_difference_maps: max_derivative_order must be >= 0, got {max_order}.")
    if max_order == 0:
        return [base]

    G = _first_derivative_operator(steps=int(base.steps), step_size=float(step_size))
    Iq = np.eye(int(base.q_dim), dtype=float)

    maps = [base]
    G_power = np.eye(int(base.steps), dtype=float)
    for _order in range(1, max_order + 1):
        G_power = G @ G_power
        D = np.kron(G_power, Iq)
        maps.append(
            TrajectoryMap(
                A=D @ base.A,
                b=D @ base.b,
                steps=int(base.steps),
                q_dim=int(base.q_dim),
            )
        )
    return maps


def _is_uniform_spacing(x: np.ndarray, *, atol: float = 1e-10, rtol: float = 1e-9) -> bool:
    if x.size < 3:
        return True
    d = np.diff(x)
    return bool(np.allclose(d, d[0], atol=atol, rtol=rtol))


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


def build_trajectory_map_with_derivative(
    traj_dsl: Mapping[str, Any],
    *,
    derivative_order: int = 0,
    derivative_wrt: str = "u",
    default_steps: int | None = None,
    default_q_dim: int | None = None,
    default_dt: float | None = None,
) -> TrajectoryMap:
    """Build one ``TrajectoryMap`` for a specified derivative order."""

    try:
        deriv_order = int(derivative_order)
    except Exception as e:
        raise ValueError(
            "build_trajectory_map_with_derivative: derivative_order must be an integer, "
            f"got {derivative_order!r}."
        ) from e
    if deriv_order < 0:
        raise ValueError(
            "build_trajectory_map_with_derivative: derivative_order must be >= 0, "
            f"got {deriv_order}."
        )

    maps = build_trajectory_maps_with_derivatives(
        traj_dsl,
        max_derivative_order=deriv_order,
        derivative_wrt=derivative_wrt,
        default_steps=default_steps,
        default_q_dim=default_q_dim,
        default_dt=default_dt,
    )
    return maps[deriv_order]


def build_trajectory_maps_with_derivatives(
    traj_dsl: Mapping[str, Any],
    *,
    max_derivative_order: int,
    derivative_wrt: str = "u",
    default_steps: int | None = None,
    default_q_dim: int | None = None,
    default_dt: float | None = None,
) -> list[TrajectoryMap]:
    """Build a ``TrajectoryMap`` for trajectory derivatives.

    Returns ``maps`` where ``maps[r]`` is the map for derivative order ``r`` with
    ``r = 0..max_derivative_order``.

    - For ``type='bspline'``:
      - ``derivative_wrt in {'u','param','parameter'}`` uses analytic derivatives.
      - ``derivative_wrt='time'`` uses analytic derivatives when parameter samples are
        uniformly spaced in time; otherwise it falls back to finite differences on
        sampled trajectory values.
    - For non-bspline types, derivatives are finite-difference approximations on the
      sampled trajectory values.
    """

    try:
        max_order = int(max_derivative_order)
    except Exception as e:
        raise ValueError(
            "build_trajectory_maps_with_derivatives: max_derivative_order must be an integer, "
            f"got {max_derivative_order!r}."
        ) from e

    if max_order < 0:
        raise ValueError(
            "build_trajectory_maps_with_derivatives: max_derivative_order must be >= 0, "
            f"got {max_order}."
        )

    if not isinstance(traj_dsl, Mapping):
        raise TypeError("build_trajectory_maps_with_derivatives: trajectory dsl must be a mapping.")

    typ = str(traj_dsl.get("type", "")).strip().lower()
    if typ == "":
        raise ValueError("build_trajectory_maps_with_derivatives: trajectory.type is required.")

    base_map = build_trajectory_map(
        traj_dsl,
        default_steps=default_steps,
        default_q_dim=default_q_dim,
    )
    if max_order == 0:
        return [base_map]

    wrt = str(derivative_wrt).strip().lower()
    if wrt in ("u", "param", "parameter"):
        fd_step_size = 1.0
    elif wrt == "time":
        if default_dt is None:
            raise ValueError(
                "build_trajectory_maps_with_derivatives: default_dt is required for derivative_wrt='time'."
            )
        fd_step_size = float(default_dt)
        if fd_step_size <= 0.0:
            raise ValueError(
                "build_trajectory_maps_with_derivatives: default_dt must be > 0 for derivative_wrt='time'. "
                f"Got {fd_step_size}."
            )
    else:
        raise ValueError(
            "build_trajectory_maps_with_derivatives: derivative_wrt must be one of "
            "'u', 'param', 'parameter', 'time'. "
            f"Got {derivative_wrt!r}."
        )

    # Non-bspline types: finite-difference approximation on sampled trajectory values.
    if typ != "bspline":
        return _finite_difference_maps(
            base_map,
            max_derivative_order=max_order,
            step_size=fd_step_size,
        )

    steps = resolve_optional_positive_int(
        pick_trajectory_value(traj_dsl, section="bspline", key="steps"),
        name="steps",
        fallback=default_steps,
    )
    q_dim = resolve_optional_positive_int(
        pick_trajectory_value(traj_dsl, section="bspline", key="q_dim"),
        name="q_dim",
        fallback=default_q_dim,
    )
    if steps is None:
        raise ValueError(
            "build_trajectory_maps_with_derivatives: steps is required for bspline trajectory "
            "(set trajectory.steps or pass default_steps)."
        )
    if q_dim is None:
        raise ValueError(
            "build_trajectory_maps_with_derivatives: q_dim is required for bspline trajectory "
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

    if wrt in ("u", "param", "parameter"):
        parameter_scale = 1.0
        return TrajectoryMap.from_bspline_derivatives(
            steps=steps,
            q_dim=q_dim,
            degree=degree,
            num_ctrl_points=num_ctrl_points,
            knot_vector=knot_vector,
            u_samples=u_samples,
            max_derivative_order=max_order,
            parameter_scale=parameter_scale,
        )

    # wrt == "time" reaches here.
    if int(steps) <= 1:
        raise ValueError(
            "build_trajectory_maps_with_derivatives: steps must be >= 2 for derivative_wrt='time'."
        )

    dt = float(fd_step_size)
    if dt <= 0.0:
        raise ValueError(
            "build_trajectory_maps_with_derivatives: default_dt must be > 0 for derivative_wrt='time'. "
            f"Got {dt}."
        )

    if knot_vector is None:
        knots = default_clamped_uniform_knots(
            num_ctrl_points=num_ctrl_points,
            degree=degree,
        )
    else:
        knots = np.asarray(knot_vector, dtype=float).reshape(-1)

    u_min = float(knots[degree])
    u_max = float(knots[num_ctrl_points])
    if u_max <= u_min:
        raise ValueError(
            "build_trajectory_maps_with_derivatives: invalid bspline knot domain for time scaling."
        )
    u_span = float(u_max - u_min)

    # If parameter samples are non-uniform, chain-rule scaling is not globally constant.
    # Fall back to finite-difference derivative maps on sampled q(t).
    if u_samples is not None:
        if u_samples.size != int(steps):
            raise ValueError(
                "build_trajectory_maps_with_derivatives: u_samples size mismatch. "
                f"Expected {steps}, got {u_samples.size}."
            )
        if not _is_uniform_spacing(u_samples):
            return _finite_difference_maps(
                base_map,
                max_derivative_order=max_order,
                step_size=dt,
            )
        u_span = float(u_samples[-1] - u_samples[0])

    horizon = float((int(steps) - 1) * dt)
    if horizon <= 0.0:
        raise ValueError(
            "build_trajectory_maps_with_derivatives: invalid time horizon for derivative_wrt='time'."
        )
    parameter_scale = u_span / horizon
    return TrajectoryMap.from_bspline_derivatives(
        steps=steps,
        q_dim=q_dim,
        degree=degree,
        num_ctrl_points=num_ctrl_points,
        knot_vector=knot_vector,
        u_samples=u_samples,
        max_derivative_order=max_order,
        parameter_scale=parameter_scale,
    )
