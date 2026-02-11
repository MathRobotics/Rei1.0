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

from eiopt import compile_problem, format_solve_report, get_named_expr_value, load_problem_toml, solve_gauss_newton
from eiopt.backends.kots import KotsTrajectoryStateBuilder
from eiopt.core.state_schema import DEFAULT_FRAME, DTYPE_KINEMATICS, make_key

_EXAMPLES_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _EXAMPLES_DIR / "models" / "planar2.json"
_ORDER = 3
_DSL_PATH = _EXAMPLES_DIR / "dsl" / "kots_traj_pos.toml"


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


def main() -> int:
    if not _MODEL_PATH.is_file():
        raise SystemExit(
            f"Model file not found: {_MODEL_PATH}\n"
            "Update `_MODEL_PATH` in examples/main_robokots_traj.py to your model JSON."
        )

    kots = Kots.from_json_file(str(_MODEL_PATH), order=_ORDER)
    data = kots.state_dict_
    dsl = load_problem_toml(_DSL_PATH)

    builder = KotsTrajectoryStateBuilder.from_dsl(kots, data, dsl=dsl)
    traj_map = builder.trajectory_map
    runtime = compile_problem(dsl, build_state=builder.build_state)

    x0 = runtime.pack.get().copy()
    x_star, _cost, _iters, _rnorm, _dxnorm, _converged = solve_gauss_newton(runtime, max_iters=3000)
    print("Optimization completed.")
    print("\tIterations:", _iters)
    print("\tFinal cost:", _cost)
    print("\tFinal residual norm:", _rnorm)
    print("\tFinal step norm:", _dxnorm)
    print("\tConverged:", _converged)

    steps = int(traj_map.steps)
    q_opt = np.vstack([traj_map.q_at(x_star, k) for k in range(steps)])
    ee_opt = _collect_ee_pos_traj(runtime, steps=steps)
    target_init = np.asarray(get_named_expr_value(runtime, name="target_pos_init"), dtype=float).reshape(-1)
    target_terminal = np.asarray(get_named_expr_value(runtime, name="target_pos_terminal"), dtype=float).reshape(-1)
    if target_init.size != 3:
        raise ValueError(f"target_pos_init size mismatch. Expected 3, got {target_init.size}.")
    if target_terminal.size != 3:
        raise ValueError(f"target_pos_terminal size mismatch. Expected 3, got {target_terminal.size}.")
    ee_target = np.vstack([target_init, target_terminal])

    print("p*:", x_star)
    for k in range(steps):
        print(f"q[{k}] =", traj_map.q_at(x_star, k))
    print("ee* (xyz):\n", ee_opt)
    print(format_solve_report(runtime, x0=x0, x_star=x_star))
    _plot_trajectory(ee_opt=ee_opt, ee_target=ee_target, q_opt=q_opt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
