from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from eiopt.backends.state.vision import VisionFieldHandler
from eiopt.backends.state.vision_pinhole import (
    PINHOLE_RADIAL_PARAM_ORDER,
    build_pinhole_radial_vision_field_handler,
    pinhole_radial_reprojection,
)
from eiopt.optimize.report import format_solve_report
from eiopt.optimize.solvers import solve
from eiopt.optimize_backends.vision import compile_camera_calibration_problem


def _build_linear_reprojection_handlers(points: np.ndarray) -> dict[str, VisionFieldHandler]:
    pts = np.asarray(points, dtype=float).reshape(-1)
    n_obs = int(pts.size)

    def reproj_value(q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
        del key, state_ref
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        if q_vec.size != 2:
            raise ValueError(
                "08_camera_calibration reproj_value(linear): expected parameter size 2 "
                f"(scale, bias), got {q_vec.size}."
            )
        scale = float(q_vec[0])
        bias = float(q_vec[1])
        return scale * pts + bias

    def reproj_jac(q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
        del key, state_ref
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        if q_vec.size != 2:
            raise ValueError(
                "08_camera_calibration reproj_jac(linear): expected parameter size 2 "
                f"(scale, bias), got {q_vec.size}."
            )
        J = np.zeros((n_obs, 2), dtype=float)
        J[:, 0] = pts
        J[:, 1] = 1.0
        return J

    return {
        "reproj": VisionFieldHandler(
            value_handler=reproj_value,
            jac_handler=reproj_jac,
        )
    }


def _build_linear_case(
    rng: np.random.Generator,
    noise_std: float,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, VisionFieldHandler]]:
    points = np.array([-1.5, -0.5, 0.5, 1.5, 2.5], dtype=float)
    theta_true = np.array([1.8, -0.3], dtype=float)  # [scale, bias]
    noise = float(noise_std) * rng.standard_normal(points.size)
    observations = theta_true[0] * points + theta_true[1] + noise

    model: dict[str, Any] = {"points": points}
    data: dict[str, Any] = {"observations": observations.copy()}
    theta_init = np.array([0.7, 1.0], dtype=float)
    handlers = _build_linear_reprojection_handlers(points)
    return model, data, observations, theta_true, theta_init, handlers


def _build_pinhole_case(
    rng: np.random.Generator,
    noise_std: float,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, VisionFieldHandler]]:
    points_xy = np.array(
        [
            [-0.45, -0.30],
            [-0.40, 0.20],
            [-0.20, -0.15],
            [-0.10, 0.35],
            [0.05, -0.25],
            [0.15, 0.10],
            [0.25, -0.05],
            [0.30, 0.30],
            [0.40, -0.20],
            [0.50, 0.15],
        ],
        dtype=float,
    )
    theta_true = np.array([700.0, 710.0, 320.0, 240.0, 0.02, -0.01], dtype=float)
    observations_clean = pinhole_radial_reprojection(theta_true, points_xy)
    noise = float(noise_std) * rng.standard_normal(observations_clean.size)
    observations = observations_clean + noise

    model: dict[str, Any] = {"points_xy": points_xy}
    data: dict[str, Any] = {"observations": observations.copy()}
    theta_init = np.array([680.0, 690.0, 300.0, 250.0, 0.0, 0.0], dtype=float)
    handlers = {
        "reproj": build_pinhole_radial_vision_field_handler(points_xy=points_xy),
    }
    return model, data, observations, theta_true, theta_init, handlers


def _build_dsl(*, observations: np.ndarray, theta_init: np.ndarray) -> dict[str, Any]:
    return {
        "vision": {
            "p_var": "theta",
            "owner_type": "camera",
            "owner_name": "cam0",
            "field": "reproj",
            "k": 0,
            "term_name": "camera_reproj_error",
            "observations": np.asarray(observations, dtype=float).reshape(-1).tolist(),
        },
        "variables": [
            {
                "name": "theta",
                "dim": int(np.asarray(theta_init, dtype=float).size),
                "init": np.asarray(theta_init, dtype=float).reshape(-1).tolist(),
            },
        ],
        "terms": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera calibration demo with dtype='vision'.")
    parser.add_argument(
        "--model",
        type=str,
        default="pinhole",
        choices=("pinhole", "linear"),
        help="Calibration model: 'pinhole' (realistic) or 'linear' (minimal mock).",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Gaussian noise std for synthetic observations.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic observations.")
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    model_kind = str(args.model)
    if model_kind == "linear":
        model, data, observations, theta_true, theta_init, handlers = _build_linear_case(rng, float(args.noise))
        max_iters = 30
    else:
        model, data, observations, theta_true, theta_init, handlers = _build_pinhole_case(rng, float(args.noise))
        max_iters = 60

    dsl = _build_dsl(observations=observations, theta_init=theta_init)

    def update_model(q: np.ndarray, model_obj: Any, data_obj: dict[str, Any]) -> None:
        del model_obj
        data_obj["theta"] = np.asarray(q, dtype=float).reshape(-1).copy()

    compiled = compile_camera_calibration_problem(
        dsl,
        model=model,
        data=data,
        field_handlers=handlers,
        update_model=update_model,
    )
    runtime = compiled.runtime

    x0 = runtime.pack.get().copy()
    x_star, initial_cost, cost, iters, rnorm, dxnorm, converged = solve(
        runtime,
        solver="gauss_newton",
        max_iters=max_iters,
        gn_damping=1e-8 if model_kind == "pinhole" else 0.0,
        gn_line_search=False,
    )

    print("=== 08_camera_calibration ===")
    print(f"model={model_kind}")
    if model_kind == "pinhole":
        print(f"param_order={list(PINHOLE_RADIAL_PARAM_ORDER.names)}")
    print(f"converged={converged} iters={iters}")
    print(
        f"cost0={initial_cost:.3e} cost={cost:.3e} "
        f"rnorm={rnorm:.3e} dxnorm={dxnorm:.3e}"
    )
    print(f"theta_true={theta_true}")
    print(f"theta_init={x0}")
    print(f"theta_est={x_star}")
    print(format_solve_report(runtime, x0=x0, x_star=x_star))


if __name__ == "__main__":
    main()
