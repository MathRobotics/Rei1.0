from __future__ import annotations

from pathlib import Path

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires matplotlib for trajectory plotting.\n"
        "Install matplotlib in your environment and re-run:\n"
        "  PYTHONPATH=. python examples/main_robokots_traj.py"
    ) from e

try:
    from robokots.kots import Kots
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires RoboKots with compatible dependencies.\n"
        "Install `robokots` and ensure `mathrobo` provides CMVector.\n"
        "Then run:\n"
        "  PYTHONPATH=. python examples/main_robokots_traj.py"
    ) from e

from eiopt import compile_problem, format_solve_report, load_problem_toml, solve_gauss_newton
from eiopt.backends.kots import KotsTrajectoryStateBuilder
from eiopt.core.trajectory import LinearTrajectoryMap
from eiopt.dsl.dsl_ops import find_const_expr

_EXAMPLES_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _EXAMPLES_DIR / "models" / "planar2.json"
_ORDER = 3
_DSL_PATH = _EXAMPLES_DIR / "dsl" / "kots_traj_pos.toml"


def _build_linear_interp_map(*, steps: int, nq: int) -> LinearTrajectoryMap:
    if steps < 2:
        raise ValueError(f"steps must be >= 2 for interpolation, got {steps}.")

    p_dim = 2 * int(nq)
    A = np.zeros((steps * nq, p_dim), dtype=float)
    b = np.zeros((steps * nq,), dtype=float)

    eye = np.eye(nq, dtype=float)
    for k in range(steps):
        alpha = float(k) / float(steps - 1)
        row = slice(k * nq, (k + 1) * nq)
        A[row, :nq] = (1.0 - alpha) * eye
        A[row, nq:] = alpha * eye

    return LinearTrajectoryMap(A=A, b=b, steps=steps, q_dim=nq)


def _collect_ee_pos_traj(runtime, *, steps: int, owner_name: str = "ee") -> np.ndarray:
    runtime.update_state_if_needed()
    ee = np.full((steps, 3), np.nan, dtype=float)

    for key in runtime.required:
        owner = getattr(key, "owner", None)
        if getattr(key, "dtype", None) != "kinematics":
            continue
        if getattr(key, "field", None) != "pos":
            continue
        if getattr(owner, "owner_type", None) != "link":
            continue
        if getattr(owner, "owner_name", None) != owner_name:
            continue
        k = int(getattr(key, "k", -1))
        if k < 0 or k >= steps:
            continue
        ee[k, :] = np.asarray(runtime.state.get(key), dtype=float).reshape(-1)[:3]

    if np.any(~np.isfinite(ee)):
        raise RuntimeError(
            f"Failed to collect complete EE trajectory for owner={owner_name!r}. "
            f"Collected={ee}"
        )
    return ee


def _load_target_pos_traj(dsl: dict, *, steps: int) -> np.ndarray:
    target_expr = find_const_expr(dsl, name="target_pos_traj")
    if target_expr is None:
        raise ValueError("DSL must contain const expr with name='target_pos_traj'.")
    target = np.asarray(target_expr.get("value", []), dtype=float).reshape(-1)
    if target.size != steps * 3:
        raise ValueError(
            "target_pos_traj length mismatch. "
            f"Expected {steps * 3} (=steps*3), got {target.size}."
        )
    return target.reshape(steps, 3)


def _plot_trajectory(*, ee_opt: np.ndarray, ee_target: np.ndarray, q_opt: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))

    ax_xy = axes[0]
    ax_xy.plot(ee_target[:, 0], ee_target[:, 1], "x--", label="target")
    ax_xy.plot(ee_opt[:, 0], ee_opt[:, 1], "o-", label="optimized")
    for k in range(ee_opt.shape[0]):
        ax_xy.text(float(ee_opt[k, 0]), float(ee_opt[k, 1]), f"k={k}", fontsize=8)
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
    ax_q.set_ylabel("q")
    ax_q.grid(True, alpha=0.35)
    ax_q.legend()

    fig.tight_layout()
    plt.show()


def main() -> int:
    if not _MODEL_PATH.is_file():
        raise SystemExit(
            f"Model file not found: {_MODEL_PATH}\n"
            "Update `_MODEL_PATH` in examples/main_robokots_traj.py to your model JSON."
        )

    kots = Kots.from_json_file(str(_MODEL_PATH), order=_ORDER)
    data = kots.state_dict_
    dsl = load_problem_toml(_DSL_PATH)

    time_dsl = dsl.get("time", {})
    steps = int(time_dsl.get("N", 0)) + 1
    nq = int(kots.dof())

    traj_map = _build_linear_interp_map(steps=steps, nq=nq)

    builder = KotsTrajectoryStateBuilder(
        kots,
        data,
        trajectory_map=traj_map,
        p_var="p",
    )
    runtime = compile_problem(dsl, build_state=builder.build_state)

    x0 = runtime.pack.get().copy()
    x_star, _cost, _iters, _rnorm, _dxnorm, _converged = solve_gauss_newton(runtime, max_iters=20)
    q_opt = np.vstack([traj_map.q_at(x_star, k) for k in range(steps)])
    ee_opt = _collect_ee_pos_traj(runtime, steps=steps, owner_name="ee")
    ee_target = _load_target_pos_traj(dsl, steps=steps)

    print("p*:", x_star)
    for k in range(steps):
        print(f"q[{k}] =", traj_map.q_at(x_star, k))
    print("ee* (xyz):\n", ee_opt)
    print(format_solve_report(runtime, x0=x0, x_star=x_star))
    _plot_trajectory(ee_opt=ee_opt, ee_target=ee_target, q_opt=q_opt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
