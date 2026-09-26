import numpy as np
import pytest

from rei.optimize.builder import compile_nls_problem
from rei.optimize.reductions import build_nullspace_equality_reduction
from rei.optimize.runtime import collect_expr_required


def state_expr(field):
    return {"type": "get_state", "name": field,
            "key": {"k": 0, "owner_type": "demo", "owner_name": "robot",
                    "dtype": "vec", "field": field}, "jac": {"var": "x"}}


def make_runtime(state_equality):
    calls = []

    def build_state(x_all, *, required=None, **kwargs):
        req = list(required)
        calls.append(req)
        x = np.asarray(x_all)
        return {key: (np.array([[1., 0.]]) if key.field.endswith("_J_x")
                      else np.array([x[0]-1.])) for key in req}

    equality = state_expr("eq_state") if state_equality else {
        "type": "sub", "a": {"type": "get_var", "var": "x"},
        "b": {"type": "const", "var": "x", "value": [1., 2.]}}
    runtime = compile_nls_problem({
        "variables": [{"name": "x", "dim": 2, "init": [0., 0.]}],
        "terms": [
            {"expr": equality, "constraint": {"kind": "eq"}, "cost": {"type": "l2"}},
            {"expr": state_expr("unused_equality"), "constraint": {"kind": "eq"}, "cost": {"type": "l2"}},
            {"expr": state_expr("torque"), "cost": {"type": "l2"}},
        ],
    }, build_state=build_state)
    return runtime, calls


@pytest.mark.parametrize("state_equality", [False, True])
@pytest.mark.parametrize("check_linearity", [False, True])
def test_reduction_only_requests_selected_equality_state(state_equality, check_linearity, monkeypatch):
    runtime, state_calls = make_runtime(state_equality)
    expected = set(collect_expr_required(runtime.problem.terms[0][0]))
    linearization_calls = []
    original = runtime.linearize_terms

    def observed(**kwargs):
        linearization_calls.append(list(kwargs["required"]))
        assert set(kwargs["required"]) == expected
        assert kwargs["term_indices"] == (0,)
        return original(**kwargs)

    monkeypatch.setattr(runtime, "linearize_terms", observed)
    reduction = build_nullspace_equality_reduction(
        runtime, eq_term_indices=[0], check_linearity=check_linearity, linearity_samples=3)
    assert len(linearization_calls) == (4 if check_linearity else 1)
    assert all(set(req) == expected for req in state_calls)
    if state_equality:
        assert state_calls
        assert {key.field for key in expected} == {"eq_state", "eq_state_J_x"}
    else:
        assert expected == set()
    assert reduction.feasibility_residual_norm < 1e-12
    np.testing.assert_allclose(reduction.constraint_jacobian @ runtime.pack.get()
                               + reduction.constraint_offset, 0., atol=1e-12)


def test_explicit_required_is_preserved_during_all_linearity_checks(monkeypatch):
    runtime, calls = make_runtime(True)
    requested = runtime.required_list()
    observed = []
    original = runtime.linearize_terms

    def linearize(**kwargs):
        observed.append(list(kwargs["required"]))
        return original(**kwargs)

    monkeypatch.setattr(runtime, "linearize_terms", linearize)
    build_nullspace_equality_reduction(runtime, eq_term_indices=[0], required=iter(requested))
    assert len(observed) == 3
    assert all(set(req) == set(requested) for req in observed + calls)
    assert any(key.field == "torque" for key in requested)


def test_no_selected_equalities_does_not_request_objective_state():
    runtime, calls = make_runtime(True)
    reduction = build_nullspace_equality_reduction(runtime, eq_term_indices=[])
    assert calls == []
    assert reduction.rank == 0
    np.testing.assert_array_equal(reduction.nullspace_basis, np.eye(2))


def test_reduced_value_evaluation_skips_derivative_state():
    runtime, calls = make_runtime(False)
    reduction = build_nullspace_equality_reduction(runtime, eq_term_indices=[0])
    reduced = reduction.runtime

    calls.clear()
    value = reduced.eval()
    assert value.shape == (2,)
    assert calls
    assert {key.field for requested in calls for key in requested} == {"unused_equality", "torque"}

    calls.clear()
    residual, jacobian = reduced.linearize()
    np.testing.assert_allclose(value, residual)
    assert jacobian.shape == (2, reduced.pack.n_total)
    assert any(key.field == "torque_J_x" for requested in calls for key in requested)


def test_trajectory_q_and_qdot_boundaries_do_not_build_torque_state():
    calls = []

    def build_state(x_all, *, required=None, **kwargs):
        calls.append(list(required))
        assert list(required) == [], "boundary reduction must not request torque state"
        return {}

    runtime = compile_nls_problem({
        "time": {"N": 4, "dt": .1},
        "trajectory": {"type": "bspline", "var": "p", "degree": 3,
                       "num_ctrl_points": 4, "q_dim": 1},
        "variables": [{"name": "p", "dim": 4, "init": [0., 0., 0., 0.]}],
        "terms": [
            {"constraint": {"kind": "eq"}, "cost": {"type": "l2"},
             "expr": {"type": "get_traj_var", "var": "p", "k": k,
                      "derivative_order": order, "derivative_wrt": "time"}}
            for k in (0, 4) for order in (0, 1)
        ] + [{"expr": {**state_expr("torque"), "jac": {"var": "p"}},
              "cost": {"type": "l2"}}],
    }, build_state=build_state)
    assert runtime.required_list()  # The full problem does require torque.
    reduction = build_nullspace_equality_reduction(runtime)
    assert reduction.rank == 4
    assert all(req == [] for req in calls)
