"""Regression coverage for state ownership, cache invalidation, and solver lifecycle."""

from copy import deepcopy

import numpy as np
import pytest

from rei import compile_nls_problem, solve
from rei.core.expr.nodes import GetStateExpr
from rei.core.expr.types import Variable, VariablePack
from rei.core.state_cache import OwnerKey, StateCache, StateKey
from rei.core.time_grid import TimeGrid
from rei.equations import SimplexMinNormProblem
from rei.optimize.costs import DiagonalWeightCost, L2Cost, ScalarWeightCost
from rei.optimize.kkt import check_kkt_residuals
from rei.optimize.runtime import NLSRuntime
from rei.problem import NLSProblem
from rei.xops import set_pack_x


def make_pack(value):
    return VariablePack([Variable('x', np.array([value], dtype=float))])


def key(frame=None):
    return StateKey(0, OwnerKey('joint', 'x'), 'coord', 'q', frame=frame)


def test_absolute_point_assignment_preserves_small_target_and_revision():
    pack = make_pack(1e16)
    target = np.array([1.])
    set_pack_x(pack, target)
    np.testing.assert_array_equal(pack.get(), target)
    assert pack.revision == 1
    target[0] = 9.
    np.testing.assert_array_equal(pack.get(), [1.])
    set_pack_x(pack, [1.])
    assert pack.revision == 1
    with pytest.raises(ValueError):
        set_pack_x(pack, [1., 2.])
    np.testing.assert_array_equal(pack.get(), [1.])
    assert pack.revision == 1


@pytest.mark.parametrize('switch', ['pack', 'time'])
def test_state_cache_distinguishes_equal_revision_objects(switch):
    calls = []
    def build(x, *, time, **kwargs):
        calls.append(1)
        return {key(): x * time.dt}
    cache = StateCache(build)
    pack, time = make_pack(1.), TimeGrid(1, 1.)
    cache.update_if_needed(pack, time=time)
    cache.update_if_needed(pack, time=time)
    assert len(calls) == 1
    if switch == 'pack':
        pack = make_pack(2.)
    else:
        time = TimeGrid(1, 2.)
    cache.update_if_needed(pack, time=time)
    np.testing.assert_array_equal(cache.get(key()), [2.])
    assert len(calls) == 2


def test_cache_invalidation_does_not_clear_provider_owned_dictionary():
    state = {key(): np.array([1.])}
    cache = StateCache(lambda *args, **kwargs: state)
    pack = make_pack(0.)
    cache.update_if_needed(pack)
    cache.invalidate()
    assert key() in state
    cache.update_if_needed(pack)
    np.testing.assert_array_equal(cache.get(key()), [1.])


def test_state_invalidation_invalidates_linearized_problem_cache():
    pack = make_pack(0.)
    value_key = key()
    jac_key = StateKey(0, value_key.owner, 'coord', 'q_J_x')
    value = [1.]
    cache = StateCache(lambda *args, **kwargs: {
        value_key: np.array(value), jac_key: np.array([value]),
    })
    expr = GetStateExpr('state', pack.vars, value_key, [jac_key])
    rt = NLSRuntime.from_problem(NLSProblem(pack, [(expr, L2Cost())]), state=cache)
    r, J = rt.linearize()
    np.testing.assert_array_equal(r, [1.])
    np.testing.assert_array_equal(J, [[1.]])
    value[0] = 2.
    cache.invalidate()
    r, J = rt.linearize()
    np.testing.assert_array_equal(r, [2.])
    np.testing.assert_array_equal(J, [[2.]])
    np.testing.assert_array_equal(rt.eval(), r)


def test_missing_state_diagnostic_handles_mixed_frame_values():
    cache = StateCache(lambda *args, **kwargs: {})
    with pytest.raises(KeyError, match='missing required keys'):
        cache.update_if_needed(make_pack(0.), required=[key(None), key('world')])


def test_stack_compilation_preserves_shared_input_key():
    inner = {'type': 'get_state', 'key': {
        'k': 0, 'owner_type': 'joint', 'owner_name': 'x', 'dtype': 'coord', 'field': 'q',
    }, 'jac': {'var': 'x'}}
    dsl = {'time': {'N': 2, 'dt': .1}, 'variables': [{'name': 'x', 'dim': 1}], 'terms': [
        {'expr': {'type': 'stack', 'range': {'k0': 0, 'k1': 2}, 'inner': inner}},
        {'expr': inner},
    ]}
    original = deepcopy(dsl)
    runtime = compile_nls_problem(dsl, build_state=lambda *args, **kwargs: {})
    assert dsl == original
    stacked = runtime.problem.terms[0][0]
    assert [part.key_value.k for part in stacked.parts] == [0, 1, 2]
    assert runtime.problem.terms[1][0].key_value.k == 0


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('field', ['grad_objective', 'eq_residual', 'ineq_residual', 'eq_jacobian', 'ineq_jacobian'])
def test_kkt_rejects_nonfinite_data(field, value):
    kwargs = dict(grad_objective=np.array([0.]), eq_residual=np.array([0.]),
                  ineq_residual=np.array([0.]), eq_jacobian=np.array([[1.]]), ineq_jacobian=np.array([[1.]]))
    kwargs[field][...] = value
    with pytest.raises(ValueError, match='finite'):
        check_kkt_residuals(**kwargs)


@pytest.mark.parametrize('field', ['active_tol', 'stationarity_tol', 'eq_tol', 'ineq_tol', 'complementarity_tol', 'dual_tol'])
def test_kkt_rejects_nan_tolerance(field):
    with pytest.raises(ValueError, match='finite'):
        check_kkt_residuals(grad_objective=[1.], **{field: np.nan})


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('cost_class', [ScalarWeightCost, DiagonalWeightCost])
def test_weighted_costs_reject_nonfinite_weights(cost_class, value):
    with pytest.raises(ValueError, match='finite'):
        cost_class(value)


def test_diagonal_weight_does_not_alias_input():
    weights = np.array([1., 2.])
    cost = DiagonalWeightCost(weights)
    weights[0] = -1.
    r, _ = cost.apply(np.ones(2), [])
    np.testing.assert_allclose(r, [1., np.sqrt(2.)])


def test_line_search_restores_point_on_evaluation_exception():
    class DomainProblem(SimplexMinNormProblem):
        def eval(self, *, required=None):
            if self.x[0] < .5:
                raise ValueError('outside domain')
            return super().eval(required=required)
    problem = DomainProblem(A=np.ones((1, 1)))
    with pytest.raises(ValueError, match='outside domain'):
        solve(problem)
    np.testing.assert_array_equal(problem.get_point(), [1.])


def test_required_generator_is_not_consumed_before_evaluation():
    requested = key()
    class RequiredProblem(SimplexMinNormProblem):
        def linearize(self, *, required=None):
            return self.eval(required=required), self.A

        def eval(self, *, required=None):
            assert tuple(required) == (requested,)
            return super().eval(required=required)
    out = solve(RequiredProblem(A=np.ones((1, 1))), required=(k for k in [requested]))
    assert out.converged


def test_trajectory_map_cache_tracks_mutated_spec_and_reuses_equal_specs():
    from rei.core.expr.registry import ExprRegister
    from rei.optimize.dsl.environment import DslBuildEnv

    env = DslBuildEnv(VariablePack([]), TimeGrid(4, .1), ExprRegister())
    spec = {'type': 'bspline', 'degree': 2, 'num_ctrl_points': 3}
    quadratic = env.resolve_trajectory_map(spec, default_q_dim=1)
    assert env.resolve_trajectory_map(dict(spec), default_q_dim=1) is quadratic
    spec['degree'] = 1
    linear = env.resolve_trajectory_map(spec, default_q_dim=1)
    assert linear is not quadratic
    assert not np.allclose(linear.A, quadratic.A)
    maps = env.resolve_trajectory_maps_with_derivatives(spec, max_derivative_order=1, default_q_dim=1)
    np.testing.assert_allclose(maps[0].A, linear.A)
    assert env.resolve_trajectory_map(spec, default_q_dim=1) is maps[0]


@pytest.mark.parametrize('method', ['projected_gradient', 'qr_nullspace'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_simplex_single_variable_rejects_nonfinite_matrix(method, value):
    from rei.equations import solve_simplex_min_norm

    with pytest.raises(ValueError, match='finite'):
        solve_simplex_min_norm([[value]], method=method)


@pytest.mark.parametrize('solver', ['gauss_newton', 'gauss_newton_no_search', 'nls'])
def test_tiny_necessary_update_is_applied_before_convergence(solver):
    from rei.optimize.solvers import nls

    if solver == 'nls':
        out = nls(lambda x: 1e15*x-1., lambda x: np.array([[1e15]]), x0=[0.])
    else:
        class ScaledProblem(SimplexMinNormProblem):
            def eval(self, *, required=None):
                return self.A @ self.x - 1.
        problem = ScaledProblem(np.array([[1e15]]))
        problem.set_point([0.])
        out = solve(problem, solver='gauss_newton', options={'line_search': solver == 'gauss_newton', 'tol_grad': 1e-10})
    assert out.converged
    np.testing.assert_allclose(out.solution, [1e-15], rtol=1e-12, atol=0.)
    assert out.stats.residual_norm < 1e-10


@pytest.mark.parametrize('line_search', [True, False])
def test_excessive_damping_is_not_mistaken_for_stationarity(line_search):
    problem = SimplexMinNormProblem(np.ones((1, 1)))
    out = solve(problem, solver='gauss_newton', options={'damping': 1e20, 'line_search': line_search})
    assert out.status == 'stalled'
    assert out.stats.residual_norm == pytest.approx(1.)
