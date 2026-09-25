from types import SimpleNamespace

import numpy as np
import pytest

from rei import compile_nls_problem_spec, prepare_noisy_ioc_trajectory, solve
from rei.core.trajectory import BsplineTrajectoryOperator, TrajectoryMap
from rei.optimize_backends.trajectory_ioc import estimate_ioc_weights


def trajectory(sparse=True):
    basis = np.array([[1., 0.], [.75, .25], [.5, .5], [0., 1.]])
    A = BsplineTrajectoryOperator(basis, 2)
    return TrajectoryMap(A if sparse else np.asarray(A), np.arange(8)*.01, 4, 2)


def test_observation_fit_matches_independent_dense_least_squares():
    mapping = trajectory()
    p = np.array([.1, .2, .3, .4])
    original = p.copy()
    result = prepare_noisy_ioc_trajectory(mapping, p, std=[.01, .03], seed=42)
    A = np.asarray(mapping.A)
    expected = np.linalg.lstsq(A, result.q_observed.reshape(-1)-mapping.b, rcond=None)[0]
    np.testing.assert_allclose(result.p, expected, atol=1e-14)
    np.testing.assert_allclose(result.q_observed-result.q_clean, result.noise, atol=1e-16)
    np.testing.assert_allclose(A.T @ (result.q_fitted-result.q_observed).reshape(-1), 0, atol=1e-14)
    np.testing.assert_array_equal(p, original)
    assert result.fit_rank == mapping.p_dim
    assert result.fit_residual_norm > 0
    dense = prepare_noisy_ioc_trajectory(trajectory(False), p, std=[.01, .03], seed=42)
    np.testing.assert_allclose(dense.p, result.p, atol=1e-14)


def test_seed_independence_zero_noise_and_no_input_aliasing():
    mapping = trajectory()
    p = np.arange(4, dtype=float)
    state = np.random.get_state()
    a = prepare_noisy_ioc_trajectory(mapping, p, std=.1, seed=4)
    b = prepare_noisy_ioc_trajectory(mapping, p, std=.1, seed=4)
    c = prepare_noisy_ioc_trajectory(mapping, p, std=.1, seed=5)
    np.testing.assert_array_equal(a.noise, b.noise)
    assert not np.array_equal(a.noise, c.noise)
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    zero = prepare_noisy_ioc_trajectory(mapping, p, std=0, seed=4)
    np.testing.assert_array_equal(zero.p, p)
    np.testing.assert_array_equal(zero.q_fitted, zero.q_clean)
    zero.p[0] = -100
    assert p[0] == 0
    selective = prepare_noisy_ioc_trajectory(mapping, p, std=[0, .1], seed=4)
    np.testing.assert_array_equal(selective.noise[:, 0], 0)


def test_rank_deficient_map_retains_original_unobserved_component():
    mapping = TrajectoryMap(np.array([[1., 0.], [1., 0.]]), np.zeros(2), 2, 1)
    out = prepare_noisy_ioc_trajectory(mapping, np.array([2., 7.]), std=.1, seed=8)
    assert out.fit_rank == 1
    assert out.p[1] == 7
    assert out.p[0] == pytest.approx(out.q_observed.mean())


@pytest.mark.parametrize("kwargs", [
    {"std": -1}, {"std": np.nan}, {"std": [1, 2, 3]}, {"std": [[1, 2]]},
    {"std": 1, "seed": -1}, {"std": 1, "seed": True}, {"std": 1, "seed": 1.5},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        prepare_noisy_ioc_trajectory(trajectory(), np.zeros(4), **kwargs)


@pytest.mark.parametrize("point", [np.zeros(3), np.zeros((2, 2)), np.full(4, np.nan)])
def test_invalid_point(point):
    with pytest.raises(ValueError, match="p must"):
        prepare_noisy_ioc_trajectory(trajectory(), point, std=.1)


def test_doc_noise_ioc_known_answer():
    def runtime(weights):
        return compile_nls_problem_spec({
            "opt_vals": {"p": {"init": [0.]}},
            "terms": [{"name": "left", "var": "p", "target": [0.], "weight": weights[0]},
                      {"name": "right", "var": "p", "target": [2.], "weight": weights[1]}],
        }, build_state=lambda *args, **kwargs: {})
    doc = solve(runtime([.75, .25]), options={"verbose": False})
    mapping = TrajectoryMap(np.ones((10, 1)), np.zeros(10), 10, 1)
    observation = prepare_noisy_ioc_trajectory(mapping, doc.solution, std=.01, seed=5)
    ioc = SimpleNamespace(runtime=runtime([1., 1.]))
    clean = estimate_ioc_weights(ioc, p=doc.solution)
    noisy = estimate_ioc_weights(ioc, p=observation.p)
    np.testing.assert_allclose(clean["weights"], [.75, .25], atol=1e-8)
    x = observation.q_observed.mean()
    np.testing.assert_allclose(noisy["weights"], [1-x/2, x/2], atol=1e-8)
    assert not np.allclose(clean["weights"], noisy["weights"])
