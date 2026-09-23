"""Regression coverage for adapter contracts and early option validation."""
from types import SimpleNamespace

import numpy as np
import pytest

from rei import as_constraint_problem, compile_nls_problem_spec, solve
from rei.optimize.solvers.gauss_newton import solve_gauss_newton
from rei.problem import NLSRuntimeConstraintProblem


def runtime():
    return compile_nls_problem_spec(
        {"opt_vals": {"x": {"init": [0.0]}}, "terms": [
            {"name": "objective", "var": "x", "target": [0.0]},
            {"name": "eq", "var": "x", "target": [1.0], "kind": "eq", "weight": 4.0},
            {"name": "ineq", "var": "x", "target": [2.0], "kind": "ineq"},
        ]}, build_state=lambda *args, **kwargs: {},
    )


def test_constraint_adapter_overrides_are_honored_without_mutation():
    original = as_constraint_problem(runtime(), kind="eq", weighted=True)
    assert as_constraint_problem(original) is original
    changed = as_constraint_problem(original, kind="ineq", weighted=False)
    np.testing.assert_allclose(original.constraint(), [-2.0])
    np.testing.assert_allclose(changed.constraint(), [-2.0])
    np.testing.assert_allclose(original.jacobian_constraint(), [[2.0]])
    np.testing.assert_allclose(changed.jacobian_constraint(), [[1.0]])
    inherited = as_constraint_problem(original, kind="ineq")
    assert inherited.weighted is True
    assert original.kind == "eq"


def test_constraint_linearization_evaluates_once_and_preserves_required_iterator():
    seen = []

    def linearize_constraint_terms(**kwargs):
        seen.append(list(kwargs["required"]))
        return [SimpleNamespace(residual=np.array([len(seen)]), jacobian=np.array([[len(seen)]]))]

    fake = SimpleNamespace(pack=SimpleNamespace(n_total=1), linearize_constraint_terms=linearize_constraint_terms)
    adapter = NLSRuntimeConstraintProblem(fake)
    r, j = adapter.linearize(required=iter(["requested"]))
    assert seen == [["requested"]]
    np.testing.assert_array_equal(r, [1])
    np.testing.assert_array_equal(j, [[1]])


@pytest.mark.parametrize("name,value", [
    ("tol_r", -1), ("tol_r", np.nan), ("tol_dx", np.inf),
    ("tol_grad", np.nan), ("damping", -1), ("damping", np.nan),
    ("damping", np.inf), ("max_iters", -1), ("max_iters", 1.5),
    ("max_iters", np.inf), ("ls_beta", 1), ("ls_beta", np.nan),
    ("ls_min_step", 0), ("ls_min_step", np.inf), ("ls_max_iters", 0),
    ("ls_max_iters", 1.5), ("ls_max_iters", np.inf),
])
@pytest.mark.parametrize("entry", ["direct", "dispatch"])
def test_invalid_options_fail_before_x0_changes_even_at_a_solution(name, value, entry):
    rt = compile_nls_problem_spec(
        {"opt_vals": {"x": {"init": [3.0]}}, "terms": [{"var": "x", "target": [0.0]}]},
        build_state=lambda *args, **kwargs: {},
    )
    with pytest.raises(ValueError, match=name):
        if entry == "direct":
            solve_gauss_newton(rt, x0=[0.0], **{name: value})
        else:
            solve(rt, x0=[0.0], options={name: value})
    np.testing.assert_array_equal(rt.pack.get(), [3.0])


def test_zero_iterations_remains_a_supported_no_step_evaluation():
    rt = runtime()
    out = solve(rt, options={"max_iters": 0})
    assert out.status == "max_iters"
    np.testing.assert_array_equal(out.solution, [0.0])
