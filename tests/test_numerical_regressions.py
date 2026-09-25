"""Regression coverage for derivatives, constrained solvers, and numerical accuracy."""

from itertools import combinations

import numpy as np
import pytest

from rei import as_solver_problem, compile_nls_problem, solve
from rei.equations import solve_simplex_min_norm
from rei.optimize.costs import HuberCost


def runtime_for(expr, variables, cost=None):
    return compile_nls_problem(
        {"variables": variables, "terms": [{"expr": expr, "cost": cost or {"type": "l2"}}]},
        build_state=lambda *_args, **_kwargs: {},
    )


def var(name):
    return {"type": "get_var", "var": name}


def sub(a, b):
    return {"type": "sub", "a": a, "b": b}


def stack(*parts):
    return {"type": "vstack", "parts": list(parts)}


def finite_difference(fn, x, eps=1e-6):
    eye = np.eye(len(x))
    return np.column_stack([(fn(x + eps * e) - fn(x - eps * e)) / (2 * eps) for e in eye])


@pytest.mark.parametrize("expr", [
    sub(var("x"), var("y")),
    stack(var("x"), var("y"), var("z")),
    sub(stack(var("x"), var("y")), stack(var("y"), var("x"))),
    stack(sub(var("x"), var("y")), sub(var("y"), var("x")), var("z")),
])
def test_composite_expression_derivatives_follow_variable_identity(expr):
    rt = runtime_for(expr, [
        {"name": "x", "init": [2., 3.]},
        {"name": "y", "init": [4., 5.]},
        {"name": "z", "init": [6.]},
    ])
    problem = as_solver_problem(rt)
    x = problem.x0
    expected = finite_difference(problem.residual, x)
    actual = problem.jacobian(x)
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    rhs = np.arange(1., actual.shape[0] + 1)
    np.testing.assert_allclose(rt.residual_vjp(rhs), expected.T @ rhs, atol=1e-8)
    expression = rt.problem.terms[0][0]
    matrix_rhs = np.column_stack([rhs, -2 * rhs])
    for variable, gradient in zip(expression.vars, expression.vjp(rt.ctx, matrix_rhs), strict=True):
        start, stop = rt.pack.slices[variable.name]
        np.testing.assert_allclose(gradient, expected[:, start:stop].T @ matrix_rhs, atol=1e-8)


def test_multivariable_subtraction_solves_coupled_problem():
    rt = runtime_for(stack(sub(var("x"), var("y")), var("y")), [
        {"name": "x", "init": [3.]}, {"name": "y", "init": [1.]},
    ])
    out = solve(rt, options={"tol_grad": 1e-10})
    assert out.converged
    np.testing.assert_allclose(out.solution, [0., 0.], atol=1e-8)


@pytest.mark.parametrize("point", [[0., 0.], [.3, .4], [1., 0.], [3., 4.], [-2., 0.]])
def test_huber_residual_jacobian_and_gradient_match_objective(point):
    rt = runtime_for(var("x"), [{"name": "x", "init": point}], {"type": "huber", "delta": 1.})
    problem = as_solver_problem(rt)
    x = problem.x0
    norm = np.linalg.norm(x)
    expected_cost = norm**2 if norm <= 1 else 2 * norm - 1
    assert problem.objective(x) == pytest.approx(expected_cost)
    expected_jac = finite_difference(problem.residual, x)
    expected_grad = finite_difference(lambda z: np.array([problem.objective(z)]), x).ravel()
    np.testing.assert_allclose(problem.jacobian(x), expected_jac, atol=1e-6)
    np.testing.assert_allclose(problem.gradient(x), expected_grad, atol=1e-6)
    residual = problem.residual(x)
    np.testing.assert_allclose(2 * rt.weighted_residual_vjp(residual), expected_grad, atol=1e-6)
    cost = HuberCost(1.)
    rhs = np.array([[1., 2., 3.], [4., 5., 6.]])
    np.testing.assert_allclose(cost.residual_vjp(x, rhs), expected_jac.T @ rhs, atol=3e-6)


def simplex_reference_objective(A):
    # Exhaust every face as an independent oracle for small test problems.
    best = np.inf
    for size in range(1, A.shape[1] + 1):
        for indices in combinations(range(A.shape[1]), size):
            face = A[:, indices]
            base = np.full(size, 1 / size)
            basis = np.vstack([np.eye(size - 1), -np.ones(size - 1)])
            z = np.linalg.lstsq(face @ basis, -face @ base, rcond=None)[0]
            weights = base + basis @ z
            if np.min(weights) >= -1e-9:
                best = min(best, .5 * np.linalg.norm(face @ weights)**2)
    return best


def assert_simplex_optimum(A):
    out = solve_simplex_min_norm(A, method="qr_nullspace")
    w = out.solution
    assert out.converged
    assert np.min(w) >= 0
    assert w.sum() == pytest.approx(1.)
    assert out.stats.objective == pytest.approx(simplex_reference_objective(A), abs=1e-8)
    g = A.T @ (A @ w)
    assert np.min(g - w @ g) >= -1e-8


def test_simplex_reconsiders_excluded_variables():
    A = np.array([[-5., 1., -4., 5.], [5., -1., 1., 3.], [-5., -3., -4., -3.]])
    assert_simplex_optimum(A)


@pytest.mark.parametrize("seed", range(20))
def test_simplex_matches_exhaustive_face_search(seed):
    rng = np.random.default_rng(seed)
    A = rng.integers(-5, 6, size=(3, 4)).astype(float)
    assert_simplex_optimum(A)


@pytest.mark.parametrize("A", [np.zeros((2, 4)), np.ones((2, 4)), np.array([[1., 1., -1., -1.]])])
def test_simplex_degenerate_faces(A):
    assert_simplex_optimum(A)


def test_simplex_iteration_limit_retains_feasible_iterate():
    A = np.array([[-5., 1., -4., 5.], [5., -1., 1., 3.], [-5., -3., -4., -3.]])
    out = solve_simplex_min_norm(A, method="qr_nullspace", max_iters=1)
    assert out.status == "max_iters"
    assert np.min(out.solution) >= 0
    assert out.solution.sum() == pytest.approx(1.)
    assert out.stats.objective <= .5 * np.linalg.norm(A @ np.full(4, .25))**2


class ScalarProblem:
    n_total = 1

    def __init__(self, x, residual, jacobian):
        self.x = np.array([x], dtype=float)
        self.residual = residual
        self.jacobian = jacobian

    def get_point(self):
        return self.x.copy()

    def set_point(self, x):
        self.x = np.array(x, dtype=float)

    def required_list(self, required=None):
        return []

    def eval(self, required=None):
        return np.atleast_1d(self.residual(self.x[0]))

    def linearize(self, required=None):
        return self.eval(), np.asarray(self.jacobian(self.x[0])).reshape(-1, 1)


def test_gauss_newton_rejected_search_is_stalled():
    problem = ScalarProblem(-1.1, lambda x: np.exp(10*x)-1, lambda x: 10*np.exp(10*x))
    with np.errstate(over="ignore"):
        out = solve(problem, solver="gauss_newton", options={"ls_max_retries": 0})
    assert out.status == "stalled"
    assert not out.converged
    np.testing.assert_array_equal(out.solution, [-1.1])
    assert out.stats.residual_norm == pytest.approx(np.linalg.norm(problem.eval()))


def test_gauss_newton_stationary_nonzero_residual_still_converges():
    problem = ScalarProblem(0., lambda x: [x-1, x+1], lambda x: [1., 1.])
    out = solve(problem)
    assert out.converged
    np.testing.assert_allclose(out.solution, [0.], atol=1e-12)


def test_kkt_dependent_inequalities_have_nonnegative_multipliers():
    from rei.optimize.kkt import check_kkt_residuals

    out = check_kkt_residuals(
        grad_objective=np.array([-1.]),
        ineq_residual=np.zeros(2),
        ineq_jacobian=np.array([[1.], [-1.]]),
    )
    assert out.ok
    assert out.stationarity_inf < 1e-12
    assert np.min(out.mu_ineq) >= 0


@pytest.mark.parametrize("seed", range(10))
def test_kkt_mixed_constraints_with_known_valid_multipliers(seed):
    from rei.optimize.kkt import check_kkt_residuals

    rng = np.random.default_rng(seed)
    eq = rng.normal(size=(2, 4))
    inequality = rng.normal(size=(6, 4))
    mu = np.maximum(rng.normal(size=6), 0.)
    gradient = -(eq.T @ rng.normal(size=2) + inequality.T @ mu)
    out = check_kkt_residuals(
        grad_objective=gradient,
        eq_residual=np.zeros(2), eq_jacobian=eq,
        ineq_residual=np.zeros(6), ineq_jacobian=inequality,
    )
    assert out.ok, out.message
    assert out.stationarity_inf < 1e-8
    assert np.min(out.mu_ineq) >= 0


def test_kkt_still_rejects_nonoptimal_feasible_point():
    from rei.optimize.kkt import check_kkt_residuals

    out = check_kkt_residuals(
        grad_objective=np.array([1.]),
        ineq_residual=np.zeros(1), ineq_jacobian=np.ones((1, 1)),
    )
    assert not out.ok
    assert out.stationarity_inf == pytest.approx(1.)


def test_nls_preserves_small_scale_direction():
    from rei.optimize.solvers import nls

    J = np.diag([1., 1e-8])
    target = J @ np.ones(2)
    out = nls(lambda x: J @ x - target, lambda x: J, x0=np.zeros(2))
    assert out.converged
    np.testing.assert_allclose(out.solution, [1., 1.], atol=1e-10)


def test_gauss_newton_applies_scale_aware_damping_floor():
    # The relative λ floor deliberately regularizes the 1e-16 diagonal entry,
    # even when the requested initial damping is zero.
    rt = runtime_for(
        sub(var("x"), {"type": "const", "var": "x", "value": [1., 1.]}),
        [{"name": "x", "init": [0., 0.]}],
        {"type": "diag_weight", "w": [1., 1e-16]},
    )
    out = solve(rt, solver="gauss_newton", options={"damping": 0., "verbose": False})
    assert out.converged
    np.testing.assert_allclose(out.solution[0], 1., atol=1e-10)
    assert 0. < out.solution[1] < .01
    assert out.history[0]["damping_min"] == pytest.approx(100. * np.finfo(float).eps)


def test_augmented_damping_preserves_regularized_step():
    # One iteration exposes the damped step, before convergence.
    rt = runtime_for(
        sub(var("x"), {"type": "const", "var": "x", "value": [2., 3.]}),
        [{"name": "x", "init": [0., 0.]}],
        {"type": "diag_weight", "w": [4., 9.]},
    )
    out = solve(rt, solver="gauss_newton", options={"damping": 2., "max_iters": 1, "line_search": False})
    np.testing.assert_allclose(out.solution, [8./6., 27./11.], atol=1e-12)
