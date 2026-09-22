"""Regression tests for constraint flags, projections, and solver adapters."""

import numpy as np
import pytest

from rei import compile_nls_problem, solve, as_linearized_problem
from rei.equations import RuntimeStationaritySource, filter_stationarity_contributions, solve_simplex_min_norm
from rei.optimize.reductions import build_nullspace_equality_reduction


def make_runtime(constraint=None):
    terms = [
        {"expr": {"type": "sub", "name": f"target_{target}",
                  "a": {"type": "get_var", "var": "x"},
                  "b": {"type": "const", "var": "x", "value": [target]}},
         "cost": {"type": "scalar_weight", "w": weight}}
        for target, weight in [(1., 1.), (3., 4.)]
    ]
    if constraint is not None:
        terms[0]["constraint"] = constraint
    return compile_nls_problem(
        {"variables": [{"name": "x", "init": [2.]}], "terms": terms},
        build_state=lambda *args, **kwargs: {},
    )


@pytest.mark.parametrize("flag", ["enabled", "is_constraint"])
@pytest.mark.parametrize("kind", ["eq", "ineq"])
def test_disabled_constraints_are_excluded_from_constraint_consumers(flag, kind):
    rt = make_runtime({"kind": kind, flag: False})
    assert rt.find_constraint_term_indices() == []
    assert rt.linearize_constraint_terms(kind=kind) == []
    reduction = build_nullspace_equality_reduction(rt)
    assert reduction.rank == 0
    assert reduction.runtime.pack.n_total == 1
    np.testing.assert_array_equal(rt.pack.get(), [2.])
    terms = RuntimeStationaritySource(rt).term_contributions()
    assert len(filter_stationarity_contributions(terms, include_constraints=False)) == 2


def test_enabled_constraint_still_reduces_variables():
    rt = make_runtime({"kind": "eq", "enabled": True})
    assert rt.find_constraint_term_indices(kind="eq") == [0]
    reduction = build_nullspace_equality_reduction(rt)
    assert reduction.rank == 1
    np.testing.assert_allclose(rt.pack.get(), [1.])


@pytest.mark.parametrize("step", [1e-20, 1e-12])
def test_tiny_projected_step_does_not_report_convergence(step):
    out = solve_simplex_min_norm([[1., 2.]], step_size=step)
    assert out.status == "stalled"
    assert out.meta["projected_gradient_norm"] > 0.1
    assert out.stats.objective > 1.


@pytest.mark.parametrize("step", [None, .1, 1e-20])
def test_projected_stationarity_accepts_boundary_optimum(step):
    out = solve_simplex_min_norm([[1., 2.]], x0=[1., 0.], step_size=step)
    assert out.converged
    assert out.stats.objective == pytest.approx(.5)
    assert out.meta["projected_gradient_norm"] < 1e-10


def test_default_projected_solver_reaches_optimum():
    out = solve_simplex_min_norm([[1., 2.]])
    assert out.converged
    np.testing.assert_allclose(out.solution, [1., 0.], atol=1e-9)


@pytest.mark.parametrize("adapted", [False, True])
def test_solve_applies_term_selection_to_runtime_and_adapter(adapted):
    rt = make_runtime()
    problem = as_linearized_problem(rt) if adapted else rt
    out = solve(problem, options={"term_indices": [0]})
    assert out.converged
    np.testing.assert_allclose(out.solution, [1.], atol=1e-8)


def test_adapter_overrides_preserve_original_and_inherit_omitted_options():
    from rei.optimize.solver_problem import as_solver_problem
    rt = make_runtime()
    adapter = as_linearized_problem(rt, weighted=False, term_indices=[1])
    inherited = as_solver_problem(adapter)
    np.testing.assert_allclose(inherited.residual([2.]), [-1.])
    overridden = as_solver_problem(adapter, weighted=True, term_indices=[0])
    np.testing.assert_allclose(overridden.residual([2.]), [1.])
    np.testing.assert_allclose(adapter.eval(), [-1.])
    out = solve(adapter)
    assert out.converged
    np.testing.assert_allclose(out.solution, [3.], atol=1e-8)


def test_adapter_weight_override_matches_raw_runtime():
    rt = make_runtime()
    adapter = as_linearized_problem(rt)
    raw = as_linearized_problem(adapter, weighted=False)
    np.testing.assert_allclose(raw.eval(), [1., -1.])
    np.testing.assert_allclose(adapter.eval(), [1., -2.])


@pytest.mark.parametrize("options", [{"term_indices": [0]}, {"weighted": False}])
def test_generic_problem_rejects_unsupported_selection_options(options):
    from rei.equations import SimplexMinNormProblem
    with pytest.raises(ValueError, match="generic LinearizedProblem"):
        solve(SimplexMinNormProblem(np.ones((1, 1))), options=options)


@pytest.mark.parametrize("scale", [1., 2e7, 1e12, 1e-12])
def test_kkt_multipliers_are_invariant_to_constraint_scaling(scale):
    from rei.optimize.kkt import check_kkt_residuals
    J = np.diag([scale, 1. / scale])
    out = check_kkt_residuals(grad_objective=[-1., -1.], ineq_residual=[0., 0.], ineq_jacobian=J)
    assert out.ok, out.message
    np.testing.assert_allclose(J.T @ out.mu_ineq, [1., 1.], atol=1e-10)
    assert np.min(out.mu_ineq) >= 0.


def test_scaled_kkt_with_equalities_and_zero_inequality_column():
    from rei.optimize.kkt import check_kkt_residuals
    out = check_kkt_residuals(
        grad_objective=[3., -1., -1.], eq_residual=[0.], eq_jacobian=[[1., 0., 0.]],
        ineq_residual=[0., 0., 0.],
        ineq_jacobian=[[0., 2e7, 0.], [0., 0., 5e-8], [0., 0., 0.]],
    )
    assert out.ok, out.message
    assert out.mu_ineq[2] == 0.


@pytest.mark.parametrize("point,expected", [
    ([1e16, 0.], [1., 0.]),
    ([1e16, 1e16], [.5, .5]),
    ([-1e16, -1e16], [.5, .5]),
    ([1e308, -1e308], [1., 0.]),
    ([1e308, 1e308, -1e308], [.5, .5, 0.]),
    ([.2, .4, .4], [.2, .4, .4]),
    ([2., 2., 1.], [.5, .5, 0.]),
])
def test_simplex_projection_is_stable_at_large_offsets(point, expected):
    from rei.equations import SimplexMinNormProblem
    problem = SimplexMinNormProblem(np.eye(len(point)), x=point)
    np.testing.assert_allclose(problem.get_point(), expected, atol=1e-14)
    projected = problem.project(point)
    assert projected.sum() == pytest.approx(1.)
    assert np.min(projected) >= 0.
    np.testing.assert_allclose(problem.project(projected), projected, atol=1e-14)


def test_simplex_projection_is_offset_invariant():
    from rei.equations import SimplexMinNormProblem
    problem = SimplexMinNormProblem(np.eye(3))
    point = np.array([.25, .5, .75])
    for offset in [-1e12, 0., 1e12]:
        np.testing.assert_allclose(problem.project(point + offset), [1/12, 1/3, 7/12], atol=1e-14)
