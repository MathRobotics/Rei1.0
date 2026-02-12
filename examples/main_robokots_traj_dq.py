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

from eiopt import format_solve_report, load_problem_toml, solve_runtime
from eiopt.backends.kots import compile_kots_trajectory_problem
from _kots_traj_common import analytic_joint_velocity, collect_ee_pos_traj, collect_target_waypoints

_EXAMPLES_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _EXAMPLES_DIR / "models" / "planar2.json"
_ORDER = 2
_DSL_PATH = _EXAMPLES_DIR / "dsl" / "kots_traj_pos_dq.toml"

# Solver selection (edit these in code)
_SOLVER = "cyipopt"  # "gauss_newton" | "scipy_minimize" | "cyipopt"
_SCIPY_METHOD = "L-BFGS-B"
_SCIPY_OPTIONS = {"maxiter": 1000}
_IPOPT_OPTIONS = {"max_iter": 1000,
                  "tol": 1e-6, "acceptable_tol": 1e-4, "acceptable_iter": 10,
                  "print_level": 5, "print_timing_statistics": "yes"}


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


def run_trajectory_dq_demo(
    *,
    solver: str = _SOLVER,
    max_iters: int = 1000,
    scipy_method: str = _SCIPY_METHOD,
    scipy_options: dict | None = None,
    ipopt_options: dict | None = None,
) -> int:
    if not _MODEL_PATH.is_file():
        raise SystemExit(
            f"Model file not found: {_MODEL_PATH}\n"
            "Update `_MODEL_PATH` in examples/main_robokots_traj_dq.py to your model JSON."
        )

    kots = Kots.from_json_file(str(_MODEL_PATH), order=_ORDER)
    data = kots.state_dict_
    dsl = load_problem_toml(_DSL_PATH)

    compiled = compile_kots_trajectory_problem(
        dsl,
        model=kots,
        data=data,
        dynamics_fields=None,
    )
    runtime = compiled.runtime
    traj_map = compiled.trajectory_map
    dt = float(compiled.dt)
    traj_dsl = dsl.get("trajectory", None)
    if not isinstance(traj_dsl, dict):
        raise SystemExit("DSL must contain [trajectory] section.")

    x0 = runtime.pack.get().copy()
    x_star, _cost, _iters, _rnorm, _dxnorm, _converged = solve_runtime(
        runtime,
        solver=solver,
        max_iters=max_iters,
        scipy_method=scipy_method,
        scipy_options=_SCIPY_OPTIONS if scipy_options is None else scipy_options,
        ipopt_options=_IPOPT_OPTIONS if ipopt_options is None else ipopt_options,
    )
    print("Optimization completed.")
    print("\tSolver:", solver)
    print("\tIterations:", _iters)
    print("\tFinal cost:", _cost)
    print("\tFinal residual norm:", _rnorm)
    print("\tFinal step norm:", _dxnorm)
    print("\tConverged:", _converged)

    steps = int(traj_map.steps)
    q_opt = np.vstack([traj_map.q_at(x_star, k) for k in range(steps)])
    qdot_opt = analytic_joint_velocity(
        x_star,
        traj_dsl=traj_dsl,
        steps=steps,
        q_dim=traj_map.q_dim,
        dt=dt,
    )
    ee_opt = collect_ee_pos_traj(runtime, steps=steps)
    target_ks, ee_target = collect_target_waypoints(dsl)

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


def main() -> int:
    return run_trajectory_dq_demo(
        solver=_SOLVER,
        max_iters=1000,
        scipy_method=_SCIPY_METHOD,
        scipy_options=_SCIPY_OPTIONS,
        ipopt_options=_IPOPT_OPTIONS,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
