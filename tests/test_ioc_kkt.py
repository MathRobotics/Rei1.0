"""Known inverse KKT solutions, signs, feasibility and conservative uniqueness."""
from types import SimpleNamespace

import numpy as np
import pytest

from rei import compile_nls_problem_spec
from rei.equations import KktConstraintData, StationarityTermContribution, estimate_kkt_weights
from rei.optimize_backends.trajectory_ioc import estimate_ioc_weights


class Gradients:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float)
        self.n_total = self.matrix.shape[0]

    def required_list(self, required=None):
        return []

    def term_contributions(self, *, required=None):
        return [StationarityTermContribution(i, f"f{i}", {}, col)
                for i, col in enumerate(self.matrix.T)]


def fit(A, *, h=(), Jh=None, g=(), Jg=None, **kwargs):
    n = np.asarray(A).shape[0]
    data = KktConstraintData(np.array(h), np.zeros((0, n)) if Jh is None else Jh,
                             np.array(g), np.zeros((0, n)) if Jg is None else Jg)
    return estimate_kkt_weights(Gradients(A), data, **kwargs)


@pytest.mark.parametrize("scaling", ["none", "gradient_norm"])
def test_equality_tangent_recovers_weights_and_free_multiplier(scaling):
    # Tangent (y) gives .5*w0-1.5*w1=0, while x requires lambda=-3.
    result = fit([[3., 3.], [.5, -1.5]], h=[0.], Jh=[[1., 0.]],
                 stationarity_scaling=scaling)
    np.testing.assert_allclose(result["weights"], [.75, .25], atol=1e-10)
    np.testing.assert_allclose(result["kkt"]["equality_multipliers"], [-3.], atol=1e-10)
    assert result["kkt"]["ok"]
    assert result["identifiability"]["unique_stationary_weights"]


def test_active_inequality_has_nonnegative_multiplier_and_inactive_is_zero():
    result = fit([[-3., -3.], [.5, -1.5]], g=[0., -2.], Jg=[[1., 0.], [0., 1.]])
    np.testing.assert_allclose(result["weights"], [.75, .25], atol=1e-9)
    np.testing.assert_allclose(result["kkt"]["inequality_multipliers"], [3., 0.], atol=1e-9)
    assert result["kkt"]["active_inequality_rows"] == [0]
    assert result["kkt"]["ok"]
    assert result["identifiability"]["unique_stationary_weights"]


def test_wrong_inequality_direction_cannot_cancel_objective():
    result = fit([[1., 2.]], g=[0.], Jg=[[1.]])
    assert not result["kkt"]["ok"]
    assert result["kkt"]["stationarity_inf"] >= 1. - 1e-10
    assert result["kkt"]["inequality_multipliers"][0] >= 0


@pytest.mark.parametrize("kind", ["eq", "ineq"])
def test_infeasible_point_is_reported_despite_exact_stationarity(kind):
    kwargs = dict(h=[.1], Jh=[[1.]]) if kind == "eq" else dict(g=[.1], Jg=[[1.]])
    result = fit([[-1., -2.]], **kwargs)
    assert result["kkt"]["stationarity_inf"] < 1e-9
    assert not result["kkt"]["ok"]
    assert not result["identifiability"]["unique_stationary_weights"]


def test_near_active_constraint_reports_complementarity_error():
    result = fit([[-100.]], g=[-1e-9], Jg=[[1.]], feasibility_tol=1e-8)
    assert result["kkt"]["complementarity_inf"] == pytest.approx(1e-7)
    assert not result["kkt"]["ok"]


@pytest.mark.parametrize("A", [[[.5, -1.5]], [[0., 0.]], [[.5, .5, -1.5]]])
def test_constraint_normal_and_duplicate_objectives_are_not_identifiable(A):
    result = fit(A, h=[0.], Jh=[[1.]])
    assert result["kkt"]["ok"]
    assert not result["identifiability"]["unique_stationary_weights"]
    assert result["identifiability"]["weight_affine_nullity"] == len(A[0]) - 1
    assert sum(result["weights"]) == pytest.approx(1.)


def test_dependent_equality_normals_do_not_destroy_weight_identification():
    result = fit([[3., 3.], [.5, -1.5]], h=[0., 0.], Jh=[[1., 0.], [2., 0.]])
    np.testing.assert_allclose(result["weights"], [.75, .25], atol=1e-10)
    assert result["identifiability"]["unique_stationary_weights"]


def test_unconstrained_case_and_zero_objective_column():
    result = fit([[.5, -1.5]])
    np.testing.assert_allclose(result["weights"], [.75, .25], atol=1e-10)
    assert result["identifiability"]["unique_stationary_weights"]
    zero = fit([[.5, -1.5, 0.]])
    assert zero["kkt"]["ok"]
    assert not zero["identifiability"]["unique_stationary_weights"]
    assert len(zero["weights"]) == 3


@pytest.mark.parametrize("kwargs", [
    {"tol": 0.}, {"active_tol": -1.}, {"feasibility_tol": np.nan},
    {"max_iters": 1.5}, {"max_iters": True}, {"max_iters": np.inf},
    {"Jh": [[1., 0.]], "h": [0.]}, {"g": [np.inf], "Jg": [[1.]]},
])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        fit([[.5, -1.5]], **kwargs)


def test_runtime_adapter_uses_signed_bounds_at_boundary_and_sets_point_first():
    runtime = compile_nls_problem_spec({
        "opt_vals": {"x": {"init": [0., 0.]}},
        "terms": [
            {"name": "f0", "var": "x", "target": [3., 0.]},
            {"name": "f1", "var": "x", "target": [3., 2.]},
            {"name": "cap", "var": "x", "bounds": {"upper": [0., 10.]}, "kind": "ineq"},
        ],
    }, build_state=lambda *args, **kwargs: {})
    result = estimate_ioc_weights(SimpleNamespace(runtime=runtime), p=[0., .5], constraint_mode="kkt")
    np.testing.assert_allclose(result["weights"], [.75, .25], atol=1e-9)
    assert result["validation"]["constraint_kkt_checked"]
    assert result["validation"]["constraint_term_indices"] == [2]
    assert result["kkt"]["ok"]
    assert result["kkt"]["inequality_multipliers"][0] == pytest.approx(3.)
    with pytest.raises(ValueError, match="incompatible"):
        estimate_ioc_weights(SimpleNamespace(runtime=runtime), constraint_mode="kkt", include_constraints=True)
    with pytest.raises(ValueError, match="constraint_mode"):
        estimate_ioc_weights(SimpleNamespace(runtime=runtime), constraint_mode="unknown")


def test_iteration_limit_does_not_claim_success():
    result = fit([[-3., -3.], [.5, -1.5]], g=[0.], Jg=[[1., 0.]], max_iters=1)
    assert result["kkt"]["status"] == "max_iters"
    assert not result["kkt"]["ok"]


def test_runtime_adapter_equality_and_forward_reconstruction():
    targets = np.array([[3., 0.], [3., 2.]])
    runtime = compile_nls_problem_spec({
        "opt_vals": {"x": {"init": [0., .5]}},
        "terms": [
            {"name": f"f{i}", "var": "x", "target": t.tolist()}
            for i, t in enumerate(targets)
        ] + [{"name": "fixed_x", "kind": "eq", "residual": {"dsl": {
            "type": "component", "base": {"type": "get_var", "var": "x"},
            "segment_dim": 2, "index": 0,
        }}}],
    }, build_state=lambda *args, **kwargs: {})
    result = estimate_ioc_weights(SimpleNamespace(runtime=runtime), constraint_mode="kkt")
    np.testing.assert_allclose(result["weights"], [.75, .25], atol=1e-9)
    # For these convex quadratics the constrained forward solution is the
    # weighted target mean projected onto x=0, an independent closed form.
    forward = np.asarray(result["weights"]) @ targets
    forward[0] = 0.
    np.testing.assert_allclose(forward, [0., .5], atol=1e-9)
    assert result["kkt"]["ok"]
