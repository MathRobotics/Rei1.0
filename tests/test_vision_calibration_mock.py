from __future__ import annotations

import pytest

from typing import Any

import numpy as np

from eiopt.backends.state.vision.provider import VisionFieldHandler
from eiopt.optimize.solvers import solve
from eiopt.optimize_backends.vision import compile_camera_calibration_problem

def _build_handlers(points: np.ndarray) -> dict[str, VisionFieldHandler]:
    pts = np.asarray(points, dtype=float).reshape(-1)
    n_obs = int(pts.size)

    def _value(q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
        del key, state_ref
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        if q_vec.size != 2:
            raise ValueError(f"expected theta size 2, got {q_vec.size}.")
        s = float(q_vec[0])
        b = float(q_vec[1])
        return s * pts + b

    def _jac(q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
        del q, key, state_ref
        J = np.zeros((n_obs, 2), dtype=float)
        J[:, 0] = pts
        J[:, 1] = 1.0
        return J

    return {
        "reproj": VisionFieldHandler(
            value_handler=_value,
            jac_handler=_jac,
        )
    }

def _dsl(*, observations: np.ndarray, init: list[float]) -> dict[str, Any]:
    return {
        "vision": {
            "p_var": "theta",
            "owner_type": "camera",
            "owner_name": "cam0",
            "field": "reproj",
            "k": 0,
            "term_name": "reproj_error",
            "observations": observations.tolist(),
        },
        "variables": [{"name": "theta", "dim": 2, "init": list(init)}],
        "terms": [],
    }

class TestVisionCalibrationMock:
    def test_compile_camera_calibration_problem_builds_runtime(self) -> None:
        points = np.array([-1.0, 0.0, 1.0], dtype=float)
        theta_true = np.array([2.0, -0.5], dtype=float)
        observations = theta_true[0] * points + theta_true[1]

        compiled = compile_camera_calibration_problem(
            _dsl(observations=observations, init=[0.0, 0.0]),
            model={"points": points},
            data={},
            p_var="theta",
            field_handlers=_build_handlers(points),
        )

        assert compiled.p_var == "theta"
        assert compiled.owner_type == "camera"
        assert compiled.owner_name == "cam0"
        assert compiled.field == "reproj"
        assert compiled.k == 0
        assert compiled.term_name == "reproj_error"
        assert compiled.n_observations == 3
        assert compiled.fields == ("reproj",)
        r, J = compiled.runtime.linearize()
        assert r.shape == (3,)
        assert J.shape == (3, 2)

    def test_compile_camera_calibration_problem_requires_owner_name_and_observations(self) -> None:
        dsl_missing_owner = {
            "vision": {"p_var": "theta", "observations": [0.0, 1.0]},
            "variables": [{"name": "theta", "dim": 2, "init": [0.0, 0.0]}],
            "terms": [],
        }
        with pytest.raises(ValueError):
            _ = compile_camera_calibration_problem(
                dsl_missing_owner,
                model={"points": np.array([0.0, 1.0], dtype=float)},
                data={},
                field_handlers=_build_handlers(np.array([0.0, 1.0], dtype=float)),
            )

        dsl_missing_observations = {
            "vision": {"p_var": "theta", "owner_name": "cam0"},
            "variables": [{"name": "theta", "dim": 2, "init": [0.0, 0.0]}],
            "terms": [],
        }
        with pytest.raises(ValueError):
            _ = compile_camera_calibration_problem(
                dsl_missing_observations,
                model={"points": np.array([0.0, 1.0], dtype=float)},
                data={},
                field_handlers=_build_handlers(np.array([0.0, 1.0], dtype=float)),
            )

    def test_compile_camera_calibration_problem_converges_to_ground_truth(self) -> None:
        points = np.array([-2.0, -0.5, 1.0, 3.0], dtype=float)
        theta_true = np.array([1.4, -0.2], dtype=float)
        observations = theta_true[0] * points + theta_true[1]
        dsl = _dsl(observations=observations, init=[0.1, 1.0])

        compiled = compile_camera_calibration_problem(
            dsl,
            model={"points": points},
            data={},
            p_var="theta",
            field_handlers=_build_handlers(points),
        )

        x_star, initial_cost, cost, _iters, _rnorm, _dxnorm, converged = solve(
            compiled.runtime,
            solver="gauss_newton",
            options={"max_iters": 16, "damping": 0.0, "line_search": False},
        )

        assert converged
        assert initial_cost > 0.0
        assert cost < 1e-20
        np.testing.assert_allclose(x_star, theta_true, rtol=0.0, atol=1e-10)
