"""Run with: uv run python developer/benchmarks/bspline_maps.py"""
from time import perf_counter
from unittest.mock import patch

import numpy as np

from rei.core import bspline
from rei.core.trajectory import TrajectoryMap


def scalar_matrix(*, u_vec, degree, knots, num_ctrl_points):
    result = np.stack([bspline.bspline_basis_row(
        u=float(u), degree=degree, knots=knots, num_ctrl_points=num_ctrl_points,
    ) for u in u_vec])
    result[np.abs(result) < 1e-14] = 0.
    return result


def measure(fn, repeats=5):
    values = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        values.append((perf_counter() - start) * 1000)
    return float(np.median(values))


def main():
    # Basis cost is independent of DOF; the operator keeps scalar basis storage.
    knots = bspline.default_clamped_uniform_knots(num_ctrl_points=20, degree=4)
    samples = np.linspace(0., 1., 100)
    def build():
        return TrajectoryMap.from_bspline_derivatives(
            steps=100, q_dim=69, degree=4, num_ctrl_points=20,
            knot_vector=knots, u_samples=samples, max_derivative_order=4)

    def cold():
        bspline._cached_derivative_basis.cache_clear()
        return build()

    with patch.object(bspline, "bspline_basis_matrix", scalar_matrix):
        baseline = measure(cold)
    cold_ms = measure(cold)
    warm_ms = measure(build)
    print("100 frames, 69 DOF, 20 controls, degree 4, derivatives 0..4")
    print(f"scalar baseline: {baseline:.3f} ms")
    print(f"vectorized cold: {cold_ms:.3f} ms")
    print(f"cached warm:     {warm_ms:.3f} ms")


if __name__ == "__main__":
    main()
