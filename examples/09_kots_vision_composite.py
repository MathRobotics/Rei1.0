from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rei.backends.state.dispatch.composite import CompositeStateBuilder
from rei.backends.state.vision.provider import CameraCalibrationStateProvider, VisionFieldHandler
from rei.core.state_schema import DTYPE_KINEMATICS, make_key
from rei.optimize.builder import compile_nls_problem
from rei.optimize.report import format_solve_report
from rei.optimize.solvers import solve

try:
    from robokots.kots import Kots
    from rei.backends.state.robotics.kots import KotsStateBuilder
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example requires RoboKots.\n"
        "Install dependencies (e.g. `uv sync --group kots`) and re-run:\n"
        "  PYTHONPATH=. python examples/09_kots_vision_composite.py"
    ) from e

_EXAMPLES_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _EXAMPLES_DIR / "models" / "planar2.json"
_ORDER = 3


def main() -> None:
    if not _MODEL_PATH.is_file():
        raise SystemExit(f"Model file not found: {_MODEL_PATH}")

    kots = Kots.from_json_file(str(_MODEL_PATH), order=_ORDER)
    data = kots.state_dict_

    kots_builder = KotsStateBuilder(
        kots,
        data,
        q_var="q",
        fields=("pos",),
        dynamics_fields=None,
    )

    H = np.array(
        [
            [1.5, -0.4],
            [-0.7, 1.2],
        ],
        dtype=float,
    )
    b = np.array([0.2, -0.1], dtype=float)

    def vision_value(q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
        del key, state_ref
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        if q_vec.size != 2:
            raise ValueError(f"vision value: expected q size 2, got {q_vec.size}.")
        return H @ q_vec + b

    def vision_jac(q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
        del q, key, state_ref
        return H.copy()

    vision_provider = CameraCalibrationStateProvider(
        model={"name": "cam0"},
        data={},
        param_var="q",
        owner_type="camera",
        field_handlers={
            "reproj": VisionFieldHandler(
                value_handler=vision_value,
                jac_handler=vision_jac,
            )
        },
    )

    composite = CompositeStateBuilder([kots_builder, vision_provider])

    q_true = np.array([0.8, -0.35], dtype=float)
    robot_key = make_key(
        k=0,
        owner_type="link",
        owner_name="ee",
        dtype=DTYPE_KINEMATICS,
        field="pos",
    )
    target_pos = np.asarray(kots_builder.build_state(q_true, required=[robot_key])[robot_key], dtype=float)
    observations = H @ q_true + b

    dsl = {
        "variables": [{"name": "q", "dim": 2, "init": [0.0, 0.0]}],
        "terms": [
            {
                "expr": {
                    "type": "sub",
                    "name": "robot_pos_error",
                    "a": {
                        "type": "get_state",
                        "key": {
                            "k": 0,
                            "owner_type": "link",
                            "owner_name": "ee",
                            "dtype": "kinematics",
                            "field": "pos",
                            "frame": "world",
                        },
                        "jac": {"var": "q"},
                    },
                    "b": {"type": "const", "var": "q", "value": target_pos.tolist()},
                },
                "cost": {"type": "l2"},
            },
            {
                "expr": {
                    "type": "sub",
                    "name": "camera_reproj_error",
                    "a": {
                        "type": "get_state",
                        "key": {
                            "k": 0,
                            "owner_type": "camera",
                            "owner_name": "cam0",
                            "dtype": "vision",
                            "field": "reproj",
                        },
                        "jac": {"var": "q"},
                    },
                    "b": {"type": "const", "var": "q", "value": observations.tolist()},
                },
                "cost": {"type": "l2"},
            },
        ],
    }

    runtime = compile_nls_problem(dsl, build_state=composite.build_state)

    x0 = runtime.pack.get().copy()
    out = solve(
        runtime,
        solver="gauss_newton",
        options={"max_iters": 40, "damping": 0.0, "line_search": False},
    )
    x_star = out.solution
    stats = out.stats

    print("=== 09_kots_vision_composite ===")
    print(f"model={_MODEL_PATH} (order={_ORDER})")
    print(f"status={stats.status} converged={stats.converged} iters={stats.iterations}")
    print(
        f"cost0={float(stats.initial_objective or 0.0):.3e} "
        f"cost={float(stats.objective or 0.0):.3e} "
        f"rnorm={float(stats.residual_norm or 0.0):.3e} "
        f"dxnorm={float(stats.step_norm or 0.0):.3e}"
    )
    print(f"q_true={q_true}")
    print(f"q_init={x0}")
    print(f"q_est={x_star}")
    print(format_solve_report(runtime, x0=x0, outcome=out))


if __name__ == "__main__":
    main()
