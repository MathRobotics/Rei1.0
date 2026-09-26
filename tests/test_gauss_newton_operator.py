"""Operator solves must work even when dense linearization is unavailable."""
import json

import numpy as np
import pytest

from rei import solve, solve_gauss_newton_operator
from rei.core.expr.nodes import GetVarExpr, ConstantExpr, SubExpr, StackExpr
from rei.core.expr.types import Variable, VariablePack, RuntimeContext
from rei.optimize.costs import L2Cost, ScalarWeightCost, DiagonalWeightCost, HuberCost
from rei.optimize.runtime import NLSRuntime
from rei.optimize.solvers._jacobian_operator import JacobianProducts, cgls_step
from rei.problem import NLSProblem, as_linearized_problem


class MatrixProblem:
    def __init__(self, A, b):
        self.A, self.b = np.asarray(A, float), np.asarray(b, float)
        self.n_total = self.A.shape[1]
        self.x = np.zeros(self.n_total)
        self.products = [0, 0]

    def get_point(self):
        return self.x.copy()

    def set_point(self, x):
        self.x = np.asarray(x, float).copy()

    def required_list(self, required=None):
        return [] if required is None else list(required)

    def eval(self, *, required=None):
        return self.A @ self.x - self.b

    def jvp(self, v, *, required=None):
        self.products[0] += 1
        return self.A @ v

    def vjp(self, w, *, required=None):
        self.products[1] += 1
        return self.A.T @ w

    def linearize(self, **kwargs):
        raise AssertionError("dense linearization is forbidden")


class NonlinearProblem(MatrixProblem):
    def __init__(self, x=.1):
        super().__init__([[1.]], [1.])
        self.x[:] = x

    def eval(self, *, required=None):
        return self.x ** 2 - 1

    def jvp(self, v, *, required=None):
        return 2 * self.x * v

    def vjp(self, w, *, required=None):
        return 2 * self.x * w


def operator_solve(problem, **options):
    return solve(problem, solver="gauss_newton_operator", options={"verbose": False, **options})


@pytest.mark.parametrize("shape", [(12, 5), (4, 9), (7, 7), (0, 3), (3, 0)])
@pytest.mark.parametrize("damping", [0., .7])
def test_cgls_matches_augmented_dense_least_squares(shape, damping):
    rng = np.random.default_rng(14)
    A, b = rng.normal(size=shape), rng.normal(size=shape[0])
    if shape == (7, 7):
        A[:, -1] = A[:, 0]  # rank deficient
    model = MatrixProblem(A, b)
    J = JacobianProducts(model, [], shape[0])
    h, info = cgls_step(J, -b, damping, tolerance=1e-11, max_iters=100)
    expected = np.linalg.lstsq(np.vstack([A, np.sqrt(damping)*np.eye(shape[1])]),
                               np.r_[b, np.zeros(shape[1])], rcond=None)[0]
    np.testing.assert_allclose(h, expected, atol=1e-9, rtol=1e-9)
    assert info["status"] == "converged"


def test_operator_only_solve_reaches_known_optimum():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(20, 6))
    b = rng.normal(size=20)
    model = MatrixProblem(A, b)
    result = operator_solve(model, tol_grad=1e-9)
    assert result.converged
    np.testing.assert_allclose(result.solution, np.linalg.lstsq(A, b, rcond=None)[0], atol=1e-9)
    assert all(model.products)
    assert result.meta["solver"] == "gauss_newton_operator"
    assert result.meta["inner_solves"]


@pytest.mark.parametrize("solver", ["gauss_newton_operator", "gauss_newton_krylov"])
def test_product_solver_verbose_replays_from_history(capsys, solver):
    from rei.optimize.history import format_solver_history

    model = MatrixProblem(np.diag([2., 3.]), [1., 1.])
    result = solve(model, solver=solver, options={"verbose": True, "max_iters": 1})
    assert any(row["event"] == "linear_solve" for row in result.history)
    output = capsys.readouterr().out
    assert "linear_solve" not in output
    assert output.rstrip() == format_solver_history(result.history).rstrip()
    assert "linear_solve" not in format_solver_history(result.history)
    assert "linear_solve" in format_solver_history(result.history, include_diagnostics=True)


def test_inner_limit_is_reported_and_outer_loop_can_continue():
    model = MatrixProblem(np.diag([1., 2., 3.]), [1., 1., 1.])
    result = operator_solve(model, max_iters=1, inner_max_iters=1)
    assert result.status == "max_iters"
    assert result.meta["inner_solves"][0]["status"] == "max_iters"
    assert result.stats.objective < result.stats.initial_objective


def test_point_bound_products_restore_trial_even_on_error():
    model = NonlinearProblem(.1)
    J = JacobianProducts(model, [], 1)
    model.set_point([2.])
    np.testing.assert_allclose(J @ np.ones(1), [.2])
    np.testing.assert_allclose(J.T @ np.ones(1), [.2])
    np.testing.assert_array_equal(model.x, [2.])
    def fail(*args, **kwargs):
        raise RuntimeError("product failure")
    model.jvp = fail
    with pytest.raises(RuntimeError, match="product failure"):
        J @ np.ones(1)
    np.testing.assert_array_equal(model.x, [2.])


def test_line_search_history_callbacks_and_dense_parity(tmp_path):
    model = NonlinearProblem()
    dense = NonlinearProblem()
    dense.linearize = lambda **kwargs: (dense.eval(), np.diag(2*dense.x))
    baseline = solve(dense, solver="gauss_newton", options={"verbose": False})
    path = tmp_path / 'history.jsonl'
    callbacks = []
    out = solve_gauss_newton_operator(model, history_vectors=True, history_path=path,
        verbose=False, on_iter=lambda k, r, dx, g: callbacks.append((k, r, dx, g)))
    assert out.converged and out.iterations == baseline.iterations
    np.testing.assert_allclose(out.solution, baseline.solution, atol=1e-10)
    assert len(callbacks) == out.iterations + 1
    trials = [e for e in out.line_search_history if e['event'] == 'line_search_trial' and e['iteration'] == 1]
    assert [e['step_scale'] for e in trials] == [1., .5, .25]
    assert [e['accepted'] for e in trials] == [False, False, True]
    assert json.loads(path.read_text().splitlines()[-1]) == out.history[-1]
    for row in out.history:
        if 'variables' in row:
            x = row['variables'][0]
            assert row['objective'] == pytest.approx((x*x-1)**2)
            assert row['jt_r'] == pytest.approx([2*x*(x*x-1)])


def test_failed_search_and_exception_restore_point(tmp_path):
    model = NonlinearProblem()
    out = operator_solve(model, ls_max_iters=1, ls_max_retries=0)
    assert out.status == 'stalled'
    np.testing.assert_array_equal(model.x, [.1])
    def fail_trial(*, required=None):
        if model.x[0] > 1:
            raise ValueError('outside domain')
        return model.x**2-1
    model.eval = fail_trial
    path = tmp_path / 'history.jsonl'
    with pytest.raises(ValueError, match='outside domain'):
        operator_solve(model, history_vectors=True, history_path=path)
    np.testing.assert_array_equal(model.x, [.1])
    assert json.loads(path.read_text().splitlines()[-1])['line_search_status'] == 'evaluation_error'


@pytest.mark.parametrize('options', [{'inner_tol': 0}, {'inner_tol': np.nan},
    {'inner_max_iters': 0}, {'inner_max_iters': 1.5}, {'max_iters': -1},
    {'ls_beta': 1}, {'damping': -1}, {'backend_options': {}}, {'unknown': 1}])
def test_invalid_options_do_not_change_point(options):
    model = NonlinearProblem(1.)
    with pytest.raises(ValueError):
        solve(model, solver='gauss_newton_operator', x0=[9.], options=options)
    np.testing.assert_array_equal(model.x, [1.])


@pytest.mark.parametrize('history', [False, True])
@pytest.mark.parametrize('max_iters', [0, 1])
def test_no_search_and_iteration_limit(history, max_iters):
    model = NonlinearProblem()
    out = operator_solve(model, history=history, max_iters=max_iters, line_search=False)
    assert out.iterations == max_iters and out.status == 'max_iters'
    assert out.line_search_history == []
    assert bool(out.history) == history


def make_runtime(cost):
    x, y = Variable('x', np.array([2., -3.])), Variable('y', np.array([.5, .7]))
    pack = VariablePack([y, x])  # global and expression order intentionally differ
    a, b = GetVarExpr('x', [x]), GetVarExpr('y', [y])
    expr = StackExpr('stack', [SubExpr('difference', a, b), a])
    runtime = NLSRuntime(NLSProblem(pack, [(expr, cost), (b, L2Cost())]), RuntimeContext(pack=pack))
    return runtime


@pytest.mark.parametrize('cost', [L2Cost(), ScalarWeightCost(3.),
    DiagonalWeightCost(np.array([1., 0., 3., 4.])), HuberCost(.5)])
@pytest.mark.parametrize('weighted', [True, False])
def test_runtime_products_match_dense_and_adjoint_without_assembly(cost, weighted, monkeypatch):
    runtime = make_runtime(cost)
    view = as_linearized_problem(runtime, weighted=weighted, term_indices=[1, 0])
    r, J = view.linearize()
    v, w = np.arange(4.)-.2, np.arange(r.size)+.1
    def forbidden(*args, **kwargs):
        raise AssertionError('dense assembly forbidden')
    monkeypatch.setattr(runtime, 'linearize_stacked_terms', forbidden)
    # All expression nodes in this runtime have dedicated products.
    for cls in (GetVarExpr, ConstantExpr, SubExpr, StackExpr):
        monkeypatch.setattr(cls, 'eval', forbidden)
    np.testing.assert_allclose(view.jvp(v), J @ v)
    np.testing.assert_allclose(view.vjp(w), J.T @ w)
    assert view.jvp(v) @ w == pytest.approx(v @ view.vjp(w))
    result = operator_solve(view, tol_grad=1e-8)
    assert result.converged


def test_runtime_term_selection_is_inherited_and_does_not_mutate_view():
    runtime = make_runtime(ScalarWeightCost(2.))
    view = as_linearized_problem(runtime, weighted=False, term_indices=[1])
    out = operator_solve(view)
    assert out.converged
    np.testing.assert_allclose(out.solution, [0., 0., 2., -3.], atol=1e-8)
    assert view.term_indices == (1,) and view.weighted is False


def test_column_norm_provider_avoids_basis_products():
    model = MatrixProblem(np.diag([2., 3.]), [1., 1.])
    model.jacobian_column_squared_norms = lambda **kwargs: np.array([4., 9.])
    J = JacobianProducts(model, [], 2)
    np.testing.assert_array_equal(J.column_squared_norms(), [4., 9.])
    assert model.products == [0, 0]


@pytest.mark.parametrize('k', [None, 1])
@pytest.mark.parametrize('sparse', [False, True])
def test_trajectory_products_exclude_affine_bias_without_dense_expansion(k, sparse, monkeypatch):
    from rei.core.trajectory import TrajectoryMap, BsplineTrajectoryOperator
    from rei.core.expr.nodes import TrajectoryVarExpr, TrajectoryVarDerivativesExpr
    p = Variable('p', np.arange(4.))
    pack = VariablePack([p])
    basis = np.array([[1., .5], [.3, 2.], [-1., .2]])
    dense = np.kron(basis, np.eye(2))
    A = BsplineTrajectoryOperator(basis, 2) if sparse else dense
    trajectory = TrajectoryMap(A, np.full(6, 1e20), 3, 2)
    expr = TrajectoryVarExpr('trajectory', [p], trajectory, k)
    multi = TrajectoryVarDerivativesExpr('derivatives', [p], [trajectory, trajectory], k)
    runtime = NLSRuntime(NLSProblem(pack, [(expr, L2Cost()), (multi, L2Cost())]), RuntimeContext(pack))
    v = np.array([.7, .2, -.3, .8])
    expected = dense @ v
    if k is not None:
        expected = expected[2*k:2*k+2]
    def forbidden(*args, **kwargs):
        raise AssertionError('dense expansion forbidden')
    monkeypatch.setattr(BsplineTrajectoryOperator, 'to_dense', forbidden)
    monkeypatch.setattr(TrajectoryVarExpr, 'eval', forbidden)
    monkeypatch.setattr(TrajectoryVarDerivativesExpr, 'eval', forbidden)
    np.testing.assert_allclose(runtime.residual_jvp(v), np.tile(expected, 3))


def test_direct_callback_local_blocks_and_custom_expression_fallback(monkeypatch):
    from rei.core.expr.types import DirectVectorExpr
    p = Variable('p', np.array([4., -2.]))
    pack = VariablePack([p])
    matrix = np.array([[1., 2.], [3., -1.], [1., .5]])
    direct = DirectVectorExpr('direct', [p], lambda ctx: matrix @ p.x,
                              lambda ctx: [matrix])
    # The fallback expression has a VJP, but no JVP capability.
    class LocalExpr:
        name, vars = 'local', [p]
        def eval(self, ctx):
            return direct.eval(ctx)
        def eval_value(self, ctx):
            return direct.eval_value(ctx)
        def vjp(self, ctx, rhs):
            return direct.vjp(ctx, rhs)
        def deps(self):
            return []
        def value_deps(self):
            return []
    runtime = NLSRuntime(NLSProblem(pack, [(direct, L2Cost()), (LocalExpr(), L2Cost())]),
                         RuntimeContext(pack))
    def forbidden(*args, **kwargs):
        raise AssertionError('global assembly forbidden')
    monkeypatch.setattr(runtime, '_assemble_global_jacobian', forbidden)
    np.testing.assert_allclose(runtime.residual_jvp([1., -1.]), np.tile(matrix @ [1., -1.], 2))
    result = operator_solve(runtime)
    assert result.converged
    np.testing.assert_allclose(result.solution, [0., 0.], atol=1e-8)


def test_roundoff_acceptance_requires_and_makes_gradient_progress():
    model = MatrixProblem([[0.], [1e-12]], [-1., -1e-12])
    out = operator_solve(model, damping=0., tol_grad=1e-25, ls_max_iters=2, ls_max_retries=0)
    assert out.converged
    np.testing.assert_allclose(out.solution, [-1.], atol=1e-10)
    assert any(e.get('reason') == 'gradient_progress' for e in out.line_search_history)


def test_search_retry_expands_budget_and_stream_only_history(tmp_path):
    path = tmp_path / 'history.jsonl'
    out = operator_solve(NonlinearProblem(), ls_max_iters=2, history=False, history_path=path)
    assert out.converged and out.history == out.line_search_history == []
    rows = [json.loads(s) for s in (tmp_path/'history.line_search.jsonl').read_text().splitlines()]
    assert any(row.get('reason') == 'expand_search' for row in rows)
    assert json.loads(path.read_text().splitlines()[-1])['status'] == 'converged'
