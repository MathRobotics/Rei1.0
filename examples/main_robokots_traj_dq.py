from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires matplotlib for trajectory plotting.\n"
        "Install matplotlib in your environment and re-run:\n"
        "  PYTHONPATH=. python examples/main_robokots_traj_dq.py"
    ) from e

try:
    from robokots.kots import Kots
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires RoboKots with compatible dependencies.\n"
        "Install `robokots` and ensure `mathrobo` provides CMVector.\n"
        "Then run:\n"
        "  PYTHONPATH=. python examples/main_robokots_traj_dq.py"
    ) from e

from eiopt import compile_problem, format_solve_report, load_problem_toml, solve_gauss_newton
from eiopt.backends.kots import KotsTrajectoryStateBuilder
from eiopt.core.state_schema import DEFAULT_FRAME, DTYPE_KINEMATICS, make_key
from eiopt.dsl.dsl_ops import find_var_dsl
from eiopt.dsl.trajectory import (
    build_trajectory_map,
    build_trajectory_map_with_derivative,
    default_steps_from_time,
)

_EXAMPLES_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _EXAMPLES_DIR / "models" / "planar2.json"
_ORDER = 3
_DSL_PATH = _EXAMPLES_DIR / "dsl" / "kots_traj_pos_dq.toml"


def _plot_trajectory(
    *,
    ee_opt: np.ndarray,
    ee_target: np.ndarray,
    target_ks: np.ndarray,
    q_opt: np.ndarray,
    qdot_opt: np.ndarray,
    dt: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    ax_xy = axes[0]
    if ee_target.shape[0] > 0:
        ax_xy.plot(ee_target[:, 0], ee_target[:, 1], "x--", label="target waypoints")
    ax_xy.plot(ee_opt[:, 0], ee_opt[:, 1], "o-", label="optimized")
    for k in range(ee_opt.shape[0]):
        ax_xy.text(float(ee_opt[k, 0]), float(ee_opt[k, 1]), f"k={k}", fontsize=8)
    for i, k in enumerate(target_ks):
        ax_xy.text(float(ee_target[i, 0]), float(ee_target[i, 1]), f"target k={int(k)}", fontsize=8, color="tab:red")
    ax_xy.set_title("End-Effector Trajectory (XY)")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.35)
    ax_xy.legend()

    ax_q = axes[1]
    ks = np.arange(q_opt.shape[0], dtype=int)
    for j in range(q_opt.shape[1]):
        ax_q.plot(ks, q_opt[:, j], "o-", label=f"q{j}")
    ax_q.set_title("Generalized Coordinates")
    ax_q.set_xlabel("k")
    ax_q.set_ylabel("q [rad]")
    ax_q.grid(True, alpha=0.35)
    ax_q.legend()

    ax_qdot = axes[2]
    ks_dot = np.arange(qdot_opt.shape[0], dtype=int)
    for j in range(qdot_opt.shape[1]):
        ax_qdot.plot(ks_dot, qdot_opt[:, j], "o-", label=f"dq{j}/dt")
    ax_qdot.set_title(f"Joint Velocity (dt={dt:g} s)")
    ax_qdot.set_xlabel("k")
    ax_qdot.set_ylabel("dq/dt [rad/s]")
    ax_qdot.grid(True, alpha=0.35)
    ax_qdot.legend()

    fig.tight_layout()
    plt.show()


def _collect_ee_pos_traj(runtime, *, steps: int) -> np.ndarray:
    required = [
        make_key(
            k=k,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
            frame=DEFAULT_FRAME,
        )
        for k in range(int(steps))
    ]
    runtime.update_state_if_needed(required=required)
    ee_list: list[np.ndarray] = []
    for key in required:
        v = np.asarray(runtime.state.get(key), dtype=float).reshape(-1)
        if v.size != 3:
            raise ValueError(f"ee pos size mismatch at k={key.k}. Expected 3, got {v.size}.")
        ee_list.append(v)
    return np.vstack(ee_list)


def _collect_target_waypoints(dsl: dict) -> tuple[np.ndarray, np.ndarray]:
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
        if key.get("owner_name") != "ee" or key.get("dtype") != "kinematics" or key.get("field") != "pos":
            continue

        k = key.get("k")
        if k is None:
            continue

        v = np.asarray(b.get("value"), dtype=float).reshape(-1)
        if v.size != 3:
            raise ValueError(f"target waypoint size mismatch at k={k}. Expected 3, got {v.size}.")
        waypoints.append((int(k), v))

    if not waypoints:
        return np.zeros((0,), dtype=int), np.zeros((0, 3), dtype=float)

    waypoints.sort(key=lambda kv: kv[0])
    target_ks = np.asarray([k for k, _v in waypoints], dtype=int)
    target_pos = np.vstack([v for _k, v in waypoints])
    return target_ks, target_pos


def _infer_model_dof(model) -> int | None:
    dof_fn = getattr(model, "dof", None)
    if callable(dof_fn):
        try:
            return int(dof_fn())
        except Exception:
            return None
    robot = getattr(model, "robot_", None)
    if robot is not None and hasattr(robot, "dof"):
        try:
            return int(getattr(robot, "dof"))
        except Exception:
            return None
    return None


def _resolve_dt(dsl: dict) -> float:
    time_dsl = dsl.get("time", {})
    if isinstance(time_dsl, dict) and "dt" in time_dsl:
        dt = float(time_dsl.get("dt"))
    else:
        dt = 1.0
    if dt <= 0.0:
        raise ValueError(f"time.dt must be > 0. Got {dt}.")
    return dt


def _analytic_joint_velocity(
    p_opt: np.ndarray,
    *,
    traj_dsl: dict,
    steps: int,
    q_dim: int,
    dt: float,
) -> np.ndarray:
    traj_d1 = build_trajectory_map_with_derivative(
        traj_dsl,
        derivative_order=1,
        derivative_wrt="time",
        default_steps=steps,
        default_q_dim=q_dim,
        default_dt=dt,
    )
    p = np.asarray(p_opt, dtype=float).reshape(-1)
    if p.size != traj_d1.p_dim:
        raise ValueError(f"parameter size mismatch: expected {traj_d1.p_dim}, got {p.size}.")
    return (traj_d1.A @ p + traj_d1.b).reshape(traj_d1.steps, traj_d1.q_dim)


def main() -> int:
    if not _MODEL_PATH.is_file():
        raise SystemExit(
            f"Model file not found: {_MODEL_PATH}\n"
            "Update `_MODEL_PATH` in examples/main_robokots_traj_dq.py to your model JSON."
        )

    kots = Kots.from_json_file(str(_MODEL_PATH), order=_ORDER)
    data = kots.state_dict_
    dsl = load_problem_toml(_DSL_PATH)

    traj_dsl = dsl.get("trajectory", None)
    if not isinstance(traj_dsl, dict):
        raise SystemExit("DSL must contain [trajectory] section.")
    p_var = str(traj_dsl.get("var", "p")).strip()
    if p_var == "":
        raise SystemExit("trajectory.var must be non-empty.")

    traj_map = build_trajectory_map(
        traj_dsl,
        default_steps=default_steps_from_time(dsl),
        default_q_dim=_infer_model_dof(kots),
    )

    p_var_dsl = find_var_dsl(dsl, name=p_var)
    if p_var_dsl is None:
        raise SystemExit(f"DSL must declare variable {p_var!r}.")
    p_dim_dsl = int(p_var_dsl.get("dim", -1))
    if p_dim_dsl != traj_map.p_dim:
        raise SystemExit(
            f"variable {p_var!r} dim mismatch: dsl={p_dim_dsl}, trajectory p_dim={traj_map.p_dim}."
        )

    builder = KotsTrajectoryStateBuilder(
        kots,
        data,
        trajectory_map=traj_map,
        p_var=p_var,
    )
    runtime = compile_problem(dsl, build_state=builder.build_state)

    x0 = runtime.pack.get().copy()
    x_star, _cost, _iters, _rnorm, _dxnorm, _converged = solve_gauss_newton(runtime, max_iters=1000)
    print("Optimization completed.")
    print("\tIterations:", _iters)
    print("\tFinal cost:", _cost)
    print("\tFinal residual norm:", _rnorm)
    print("\tFinal step norm:", _dxnorm)
    print("\tConverged:", _converged)

    dt = _resolve_dt(dsl)
    steps = int(traj_map.steps)
    q_opt = np.vstack([traj_map.q_at(x_star, k) for k in range(steps)])
    qdot_opt = _analytic_joint_velocity(
        x_star,
        traj_dsl=traj_dsl,
        steps=steps,
        q_dim=traj_map.q_dim,
        dt=dt,
    )
    ee_opt = _collect_ee_pos_traj(runtime, steps=steps)
    target_ks, ee_target = _collect_target_waypoints(dsl)

    print("p*:", x_star)
    for k in range(steps):
        print(f"q[{k}] =", traj_map.q_at(x_star, k))
    for k in range(qdot_opt.shape[0]):
        print(f"qdot[{k}] =", qdot_opt[k])
    print("ee* (xyz):\n", ee_opt)
    print(format_solve_report(runtime, x0=x0, x_star=x_star))
    _plot_trajectory(
        ee_opt=ee_opt,
        ee_target=ee_target,
        target_ks=target_ks,
        q_opt=q_opt,
        qdot_opt=qdot_opt,
        dt=dt,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
