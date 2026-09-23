"""Signed margins for KKT must not inherit the zero slope of hinge penalties."""
import numpy as np
import pytest

from rei import compile_nls_problem_spec
from rei.optimize.builder import compile_nls_problem
from rei.optimize.kkt import check_kkt_conditions


def runtime_at(x, target, bounds, weight=100.0):
    return compile_nls_problem_spec(
        {"opt_vals": {"x": {"init": [x]}}, "terms": [
            {"name": "objective", "var": "x", "target": [target]},
            {"name": "limits", "var": "x", "bounds": bounds, "kind": "ineq", "weight": weight},
        ]}, build_state=lambda *args, **kwargs: {},
    )


@pytest.mark.parametrize("weight", [1.0, 1e5])
@pytest.mark.parametrize("bounds,x,target,expected_j", [
    ({"upper": [1.0]}, 1.0, 2.0, 1.0),
    ({"lower": [1.0]}, 1.0, 0.0, -1.0),
])
def test_boundary_optimum_passes_kkt_without_changing_penalty(bounds, x, target, expected_j, weight):
    runtime = runtime_at(x, target, bounds, weight)
    before = runtime.linearize()
    margin, jacobian = runtime.linearize_inequality_constraints()
    np.testing.assert_allclose(margin, [0.0])
    np.testing.assert_allclose(jacobian, [[expected_j]])
    result = check_kkt_conditions(runtime)
    assert result.ok, result.message
    assert result.n_active_ineq_rows == 1
    np.testing.assert_allclose(result.mu_ineq, [1.0])
    for old, new in zip(before, runtime.linearize()):
        np.testing.assert_array_equal(old, new)
    penalty = runtime.linearize_terms(term_indices=[1], weighted=False)[0]
    np.testing.assert_array_equal(penalty.jacobian, [[0.0]])


@pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
def test_two_sided_bounds_preserve_order_and_sign(x):
    runtime = runtime_at(x, x, {"lower": [-1.0], "upper": [1.0]})
    margins, jacobian = runtime.linearize_inequality_constraints()
    np.testing.assert_allclose(margins, [x - 1, -1 - x])
    np.testing.assert_allclose(jacobian, [[1.0], [-1.0]])
    result = check_kkt_conditions(runtime)
    assert result.ok
    assert result.n_active_ineq_rows == (0 if x == 0 else 1)


def test_interior_cannot_supply_a_constraint_multiplier():
    runtime = runtime_at(0.0, 2.0, {"upper": [1.0]})
    result = check_kkt_conditions(runtime)
    assert not result.ok
    assert result.n_active_ineq_rows == 0
    assert result.stationarity_inf == 2.0
    np.testing.assert_array_equal(result.mu_ineq, [0.0])


@pytest.mark.parametrize("x", [-2.0, 2.0])
def test_violation_is_not_hidden(x):
    runtime = runtime_at(x, x, {"lower": [-1.0], "upper": [1.0]})
    result = check_kkt_conditions(runtime)
    assert not result.ok
    assert result.ineq_violation_inf == 1.0


def test_signed_greater_equal_remains_supported():
    runtime = compile_nls_problem(
        {"variables": [{"name": "x", "init": [0.0]}], "terms": [
            {"expr": {"type": "sub", "a": {"type": "get_var", "var": "x"},
                      "b": {"type": "const", "var": "x", "value": [-1.0]}}, "cost": {"type": "l2"}},
            {"expr": {"type": "get_var", "var": "x"}, "constraint": {"kind": "ineq"}, "cost": {"type": "l2"}},
        ]}, build_state=lambda *args, **kwargs: {},
    )
    assert check_kkt_conditions(runtime, ineq_sense=">=").ok
    assert not check_kkt_conditions(runtime, ineq_sense="<=").ok


def test_bounds_reject_conflicting_sense():
    runtime = runtime_at(1.0, 2.0, {"upper": [1.0]})
    with pytest.raises(ValueError, match="Hinge/bounds"):
        check_kkt_conditions(runtime, ineq_sense=">=")


def test_empty_selection_has_full_variable_width():
    runtime = runtime_at(1.0, 2.0, {"upper": [1.0]})
    r, j = runtime.linearize_inequality_constraints(term_indices=[])
    assert r.shape == (0,)
    assert j.shape == (0, 1)


def test_nested_stacks_align_multiple_variable_blocks():
    from rei.core.expr.nodes import StackExpr

    runtime = compile_nls_problem_spec(
        {"opt_vals": {"x": {"init": [1.0]}, "y": {"init": [-2.0]}},
         "terms": [
             {"var": "x", "kind": "ineq", "bounds": {"lower": [-1.0], "upper": [1.0]}},
             {"var": "y", "kind": "ineq", "bounds": {"lower": [-2.0]}},
         ]}, build_state=lambda *args, **kwargs: {},
    )
    first, cost = runtime.problem.terms[0]
    second, _ = runtime.problem.terms[1]
    runtime.problem.terms[0] = (StackExpr(name="nested", parts=[first, second]), cost)
    r, j = runtime.linearize_inequality_constraints(term_indices=[0])
    np.testing.assert_allclose(r, [0.0, -2.0, 0.0])
    np.testing.assert_allclose(j, [[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]])
