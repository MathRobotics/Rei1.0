import numpy as np
import pytest

from rei.core.bspline import (
    bspline_basis_derivative_matrix,
    bspline_basis_derivative_matrices,
    default_clamped_uniform_knots,
)
from rei.core.trajectory import TrajectoryMap
from rei.core.trajectory_dsl import build_trajectory_maps_with_derivatives


@pytest.mark.parametrize("degree", [0, 1, 3])
def test_basis_rejects_derivative_above_degree(degree):
    kwargs = dict(u_vec=np.linspace(0, 1, 5), degree=degree,
                  num_ctrl_points=degree+2,
                  knots=default_clamped_uniform_knots(num_ctrl_points=degree+2, degree=degree))
    match = rf"derivative order {degree+1} > degree {degree}"
    with pytest.raises(ValueError, match=match):
        bspline_basis_derivative_matrix(**kwargs, derivative_order=degree+1)
    with pytest.raises(ValueError, match=match):
        bspline_basis_derivative_matrices(**kwargs, max_derivative_order=degree+1)
    assert bspline_basis_derivative_matrices(**kwargs, max_derivative_order=degree).shape == (degree+1, 5, degree+2)


@pytest.mark.parametrize("degree", [0, 1, 3])
def test_trajectory_map_rejects_derivative_above_degree(degree):
    kwargs = dict(steps=5, q_dim=2, degree=degree, num_ctrl_points=degree+2)
    with pytest.raises(ValueError, match="requested derivative order must be <= degree"):
        TrajectoryMap.from_bspline_derivative(**kwargs, derivative_order=degree+1)
    with pytest.raises(ValueError, match="requested derivative order must be <= degree"):
        TrajectoryMap.from_bspline_derivatives(**kwargs, max_derivative_order=degree+1)
    assert len(TrajectoryMap.from_bspline_derivatives(**kwargs, max_derivative_order=degree)) == degree+1


@pytest.mark.parametrize("wrt", ["time", "u"])
@pytest.mark.parametrize("samples", [None, [0., .1, .3, .6, 1.]])
def test_dsl_does_not_bypass_limit_with_nonuniform_samples(wrt, samples):
    spec = dict(type="bspline", degree=2, num_ctrl_points=4, steps=5, q_dim=1)
    if samples is not None:
        spec["u_samples"] = samples
    with pytest.raises(ValueError, match="derivative order 3 > degree 2"):
        build_trajectory_maps_with_derivatives(spec, max_derivative_order=3,
                                              derivative_wrt=wrt, default_dt=.1)
    assert len(build_trajectory_maps_with_derivatives(spec, max_derivative_order=2,
                                                    derivative_wrt=wrt, default_dt=.1)) == 3
