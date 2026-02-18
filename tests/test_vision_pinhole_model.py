from __future__ import annotations

import unittest

import numpy as np

from eiopt.backends.state.vision_pinhole import (
    PINHOLE_RADIAL_PARAM_ORDER,
    build_pinhole_radial_vision_field_handler,
    pinhole_radial_reprojection,
    pinhole_radial_reprojection_jacobian,
)
from eiopt.optimize.solvers import solve
from eiopt.optimize_backends.vision import compile_camera_calibration_problem


def _points_xy() -> np.ndarray:
    return np.array(
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


class TestVisionPinholeModel(unittest.TestCase):
    def test_pinhole_radial_reprojection_shapes(self) -> None:
        pts = _points_xy()
        theta = np.array([700.0, 710.0, 320.0, 240.0, 0.01, -0.003], dtype=float)
        y = pinhole_radial_reprojection(theta, pts)
        J = pinhole_radial_reprojection_jacobian(theta, pts)

        self.assertEqual(theta.size, PINHOLE_RADIAL_PARAM_ORDER.size)
        self.assertEqual(y.shape, (2 * pts.shape[0],))
        self.assertEqual(J.shape, (2 * pts.shape[0], PINHOLE_RADIAL_PARAM_ORDER.size))

    def test_pinhole_radial_reprojection_jacobian_matches_finite_difference(self) -> None:
        pts = _points_xy()
        theta = np.array([620.0, 615.0, 300.0, 220.0, 0.015, -0.007], dtype=float)
        J = pinhole_radial_reprojection_jacobian(theta, pts)

        y0 = pinhole_radial_reprojection(theta, pts)
        J_fd = np.zeros_like(J)
        eps = 1e-7
        for i in range(theta.size):
            h = eps * max(1.0, abs(float(theta[i])))
            qp = theta.copy()
            qp[i] += h
            yp = pinhole_radial_reprojection(qp, pts)
            J_fd[:, i] = (yp - y0) / h

        np.testing.assert_allclose(J, J_fd, rtol=1e-6, atol=1e-6)

    def test_pinhole_radial_calibration_converges(self) -> None:
        pts = _points_xy()
        theta_true = np.array([700.0, 710.0, 320.0, 240.0, 0.02, -0.01], dtype=float)
        observations = pinhole_radial_reprojection(theta_true, pts)

        dsl = {
            "vision": {
                "p_var": "theta",
                "owner_type": "camera",
                "owner_name": "cam0",
                "field": "reproj",
                "k": 0,
                "term_name": "camera_reproj_error",
                "observations": observations.tolist(),
            },
            "variables": [
                {"name": "theta", "dim": 6, "init": [680.0, 690.0, 300.0, 250.0, 0.0, 0.0]},
            ],
            "terms": [],
        }

        compiled = compile_camera_calibration_problem(
            dsl,
            model={"points_xy": pts},
            data={},
            field_handlers={"reproj": build_pinhole_radial_vision_field_handler(points_xy=pts)},
        )

        x_star, _cost0, cost, _iters, _rnorm, _dxnorm, converged = solve(
            compiled.runtime,
            solver="gauss_newton",
            max_iters=40,
            gn_damping=1e-8,
        )
        self.assertTrue(converged)
        self.assertLess(cost, 1e-16)
        np.testing.assert_allclose(x_star, theta_true, rtol=0.0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
