from __future__ import annotations

from pathlib import Path

import numpy as np

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

    print("p*:", x_star)
    for k in range(steps):
        print(f"q[{k}] =", traj_map.q_at(x_star, k))
    print(format_solve_report(runtime, x0=x0, x_star=x_star))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
