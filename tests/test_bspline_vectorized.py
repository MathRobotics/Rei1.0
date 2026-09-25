import numpy as np
import pytest

from rei.core import bspline
from rei.core.trajectory import TrajectoryMap


@pytest.mark.parametrize("degree", range(6))
def test_vectorized_basis_matches_scalar_recurrence(degree):
    n = degree + 7
    knots = bspline.default_clamped_uniform_knots(num_ctrl_points=n, degree=degree)
    # Include endpoints, exact interior knots, and adjacent floating point values.
    u = np.unique(np.r_[np.linspace(0, 1, 100), knots,
                        np.nextafter(knots, 0), np.nextafter(knots, 1)])
    actual = bspline.bspline_basis_matrix(u_vec=u, degree=degree, knots=knots, num_ctrl_points=n)
    expected = np.stack([bspline.bspline_basis_row(u=t, degree=degree, knots=knots,
                                                  num_ctrl_points=n) for t in u])
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-13)


def test_shifted_scaled_windows_reuse_basis_and_preserve_coefficients():
    bspline._cached_derivative_basis.cache_clear()
    knots = np.array([0., 0., 0., 0., .25, .5, .5, .75, 1., 1., 1., 1.])
    samples = np.linspace(0, 1, 33)
    def maps(knots, samples, scale):
        return TrajectoryMap.from_bspline_derivatives(
            steps=33, q_dim=2, degree=3, num_ctrl_points=8,
            knot_vector=knots, u_samples=samples, max_derivative_order=3,
            parameter_scale=scale)
    original = maps(knots, samples, 1.)
    before = bspline._cached_derivative_basis.cache_info()
    shifted = maps(8. * knots + 32., 8. * samples + 32., 8.)
    after = bspline._cached_derivative_basis.cache_info()
    assert after.misses == before.misses
    assert after.hits - before.hits == 4
    p = np.random.default_rng(0).normal(size=16)
    for order in range(4):
        np.testing.assert_allclose(shifted[order].A @ p, original[order].A @ p,
                                   atol=1e-11, rtol=1e-12)
    # Returned operators must not expose mutable cached storage.
    original[0].A.basis[:] = 0
    again = maps(knots, samples, 1.)
    np.testing.assert_allclose(again[0].A.basis, shifted[0].A.basis)


def test_global_knots_with_moving_samples_do_not_reuse_wrong_window():
    knots = bspline.default_clamped_uniform_knots(num_ctrl_points=8, degree=3)
    p = np.arange(8., dtype=float)
    results = []
    for samples in (np.linspace(.1, .4, 20), np.linspace(.4, .7, 20)):
        maps = TrajectoryMap.from_bspline_derivatives(
            steps=20, q_dim=1, degree=3, num_ctrl_points=8,
            knot_vector=knots, u_samples=samples, max_derivative_order=3)
        reference = bspline._derivative_matrices_uncached(
            u_vec=samples, degree=3, knots=knots, num_ctrl_points=8, orders=set(range(4)))
        for order, trajectory in enumerate(maps):
            np.testing.assert_allclose(trajectory.A @ p, reference[order] @ p, atol=1e-10)
        results.append(maps[0].A @ p)
    assert not np.allclose(*results)


def test_parameter_derivatives_scale_with_absolute_knot_domain():
    knots = 4. * bspline.default_clamped_uniform_knots(num_ctrl_points=8, degree=3) + 12.
    samples = np.linspace(12., 16., 30)
    maps = TrajectoryMap.from_bspline_derivatives(
        steps=30, q_dim=1, degree=3, num_ctrl_points=8,
        knot_vector=knots, u_samples=samples, max_derivative_order=3)
    reference = bspline._derivative_matrices_uncached(
        u_vec=samples, degree=3, knots=knots, num_ctrl_points=8, orders=set(range(4)))
    for order, trajectory in enumerate(maps):
        np.testing.assert_allclose(trajectory.A.basis, reference[order], atol=1e-11)
