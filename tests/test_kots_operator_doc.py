"""Real Rust DOC product routing, plus reduced and fallback product contracts."""
from pathlib import Path
import copy
import numpy as np
import pytest

from rei import solve
from rei.problem import as_linearized_problem, NLSProblem
from rei.core.expr.nodes import GetStateExpr
from rei.core.expr.types import Variable, VariablePack, RuntimeContext
from rei.core.state_cache import StateCache, StateKey, OwnerKey
from rei.optimize.costs import L2Cost
from rei.optimize.runtime import NLSRuntime


@pytest.fixture
def kots_doc(request):
    Kots = pytest.importorskip('robokots.kots').Kots
    if not hasattr(Kots, 'from_urdf_file'):
        pytest.skip('Native-model RoboKots is required.')
    from rei import load_problem_spec_toml
    from rei.optimize_backends.kots import compile_kots_trajectory_problem
    from rei.optimize.reductions import build_nullspace_equality_reduction
    root = Path(__file__).resolve().parents[1]
    spec = load_problem_spec_toml(root/'examples/spec/robokots_traj_dynamics_d12.toml')
    spec['time'].update(N=20, dt=.1)
    spec['trajectory']['num_ctrl_points'] = 8
    field, order = getattr(request, 'param', ('torque', 3))
    template = spec['terms'].pop()
    for selected in (field if isinstance(field, tuple) else (field,)):
        term = copy.deepcopy(template)
        term['expr']['inner']['key']['field'] = selected
        spec['terms'].append(term)
    model = Kots.from_urdf_file(str(root/'examples/models/planar2.urdf'), order=order)
    compiled = compile_kots_trajectory_problem(spec, model=model, kots_backend='rust',
        jacobian_strategy='mul', gravity=(0., 0., -9.81))
    reduction = build_nullspace_equality_reduction(compiled.runtime,
        eq_selector_attr='enforce', eq_selector_value='nullspace')
    return compiled, reduction, model


@pytest.mark.parametrize('kots_doc,solver', [
    (('torque', 3), 'gauss_newton_operator'),
    (('torque', 3), 'gauss_newton_krylov'),
    (('torque_d1', 4), 'gauss_newton_krylov'),
    (('torque_d2', 5), 'gauss_newton_krylov'),
], indirect=['kots_doc'])
def test_rust_doc_reduced_products_are_adjoint_and_avoid_dense_jacobians(kots_doc, monkeypatch, solver):
    compiled, reduction, model = kots_doc
    runtime = reduction.runtime
    rng = np.random.default_rng(32)
    runtime.pack.set(rng.normal(size=runtime.pack.n_total)*.02)
    r, J = runtime.linearize()
    v, w = rng.normal(size=J.shape[1]), rng.normal(size=J.shape[0])
    native_jvp, native_vjp = model.jacobian_mul, model.jacobian_transpose_mul
    calls = []
    def jvp(ref, rhs, *args, **kwargs):
        assert rhs.ndim == 3 and rhs.shape[-1] in (1, rhs.shape[1])
        calls.append('jvp')
        return native_jvp(ref, rhs, *args, **kwargs)
    def vjp(*args, **kwargs):
        calls.append('vjp')
        return native_vjp(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('dense Jacobian generation forbidden')
    monkeypatch.setattr(model, 'jacobian_mul', jvp)
    monkeypatch.setattr(model, 'jacobian_transpose_mul', vjp)
    native_many = getattr(model, 'jacobian_transpose_mul_many', None)
    if callable(native_many):
        def vjp_many(*args, **kwargs):
            calls.append('vjp')
            return native_many(*args, **kwargs)
        monkeypatch.setattr(model, 'jacobian_transpose_mul_many', vjp_many)
    monkeypatch.setattr(model, 'jacobian', forbidden)
    monkeypatch.setattr(compiled.runtime, '_assemble_global_jacobian', forbidden)
    monkeypatch.setattr(GetStateExpr, 'eval', forbidden)
    compiled.runtime.state.invalidate()
    view = as_linearized_problem(runtime)
    req = view.operator_required_list()
    assert not any('_J_' in key.field for key in req)
    actual_jv, actual_jtw = view.jvp(v, required=req), view.vjp(w, required=req)
    np.testing.assert_allclose(actual_jv, J @ v, rtol=2e-10, atol=1e-10)
    np.testing.assert_allclose(actual_jtw, J.T @ w, rtol=2e-10, atol=1e-10)
    assert w @ actual_jv == pytest.approx(v @ actual_jtw, abs=1e-10)
    out = solve(view, solver=solver, options={'verbose':False, 'tol_grad':1e-8})
    assert out.converged
    assert 'jvp' in calls and 'vjp' in calls


def test_batched_jvp_keeps_request_order_and_independent_directions(kots_doc):
    compiled, reduction, _model = kots_doc
    full = compiled.runtime
    # Pick the trajectory state cache's bound backend.
    builder = full.state.build_state.__self__
    keys = [k for k in full.operator_required_list() if k.dtype == 'dynamics']
    chosen = [keys[-1], keys[0], keys[-1]]
    rng = np.random.default_rng(53)
    directions = rng.normal(size=(len(chosen), full.pack.n_total))
    expected=[]
    for key, v in zip(chosen, directions, strict=True):
        jac_key=StateKey(key.k, key.owner, key.dtype, key.field+'_J_'+builder.p_var, key.frame, key.rel_frame)
        expected.append(builder.build_state(full.pack.get(), pack=full.pack, time=full.time,
                                            required=[jac_key])[jac_key] @ v)
    products=builder.param_jacobian_mul_many(full.pack.get(), list(zip(chosen,directions)),
                                            pack=full.pack, time=full.time)
    np.testing.assert_allclose(products, expected, rtol=2e-10, atol=1e-8)


@pytest.mark.parametrize('kots_doc', [(('torque_d2', 'torque', 'torque_d1'), 5)], indirect=True)
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('subset', [False, True])
def test_three_torque_fields_share_one_native_jvp(kots_doc, monkeypatch, weighted, subset):
    compiled, reduction, model = kots_doc
    runtime = reduction.runtime
    rng = np.random.default_rng(87)
    runtime.pack.set(rng.normal(size=runtime.pack.n_total)*.01)
    selection = list(reversed(runtime.objective_term_indices[-2:])) if subset else None
    _, J = runtime.linearize_stacked_terms(weighted=weighted, term_indices=selection)
    v = rng.normal(size=J.shape[1])
    original = model.jacobian_mul
    calls = []
    reject_shared = False
    def counted(*args, **kwargs):
        calls.append(1)
        if reject_shared and len(args[0]) > 2:  # planar2: two rows per field
            raise NotImplementedError('Single field only')
        return original(*args, **kwargs)
    monkeypatch.setattr(model, 'jacobian_mul', counted)
    actual = runtime.residual_jvp(v, weighted=weighted, term_indices=selection)
    np.testing.assert_allclose(actual, J @ v, rtol=2e-10, atol=1e-8)
    assert len(calls) == 1
    reject_shared = True
    calls.clear()
    fallback = runtime.residual_jvp(v, weighted=weighted, term_indices=selection)
    np.testing.assert_allclose(fallback, J @ v, rtol=2e-10, atol=1e-8)
    assert len(calls) == (3 if subset else 4)
    reject_shared = False
    # Different directions at matching frames must not be merged.
    full = compiled.runtime
    builder = full.state.build_state.__self__
    keys = [k for k in full.operator_required_list() if k.dtype == 'dynamics' and k.k == 0]
    directions = rng.normal(size=(len(keys), full.pack.n_total))
    requests = list(zip(keys, directions))
    expected = [builder.param_jacobian_mul_many(full.pack.get(), [request], pack=full.pack,
                                              time=full.time)[0] for request in requests]
    calls.clear()
    actual = builder.param_jacobian_mul_many(full.pack.get(), requests, pack=full.pack, time=full.time)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-8)
    assert len(calls) == len(keys)


def test_operator_value_dependencies_allow_dense_backend_fallback():
    x = Variable('x', np.array([.5]))
    pack = VariablePack([x])
    key = StateKey(0, OwnerKey('joint','test'), 'coord','q')
    jac = StateKey(0,key.owner,'coord','q_J_x')
    requested=[]
    def build(values, *, required, **kwargs):
        requested.append(set(required))
        return {k: np.array([values[0]**2-1]) if k==key else np.array([[2*values[0]]])
                for k in required}
    state=StateCache(build)
    runtime=NLSRuntime(NLSProblem(pack, [(GetStateExpr('state',[x],key,[jac]), L2Cost())]),
                       RuntimeContext(pack,state))
    result=solve(runtime,solver='gauss_newton_operator',options={'verbose':False})
    assert result.converged
    assert result.solution == pytest.approx([1.])
    assert requested[0] == {key}
    assert any(jac in keys for keys in requested)


@pytest.mark.parametrize('kots_doc', [(('torque_d2', 'torque', 'torque_d1'), 5)], indirect=True)
@pytest.mark.parametrize('matrix_rhs', [True, False])
def test_local_motion_cache_preserves_products_and_invalidates(kots_doc, monkeypatch, matrix_rhs):
    compiled, reduction, model = kots_doc
    runtime = reduction.runtime
    builder = compiled.runtime.state.build_state.__self__
    rng = np.random.default_rng(734)
    runtime.pack.set(rng.normal(size=runtime.pack.n_total)*.03)
    _, J = runtime.linearize()
    v, w = rng.normal(size=J.shape[1]), rng.normal(size=J.shape[0])
    original = model.jacobian_mul
    calls = []
    def counted(refs, rhs):
        calls.append(rhs.shape[-1])
        if not matrix_rhs and rhs.shape[-1] != 1:
            raise NotImplementedError('Vector RHS only')
        return original(refs, rhs)
    monkeypatch.setattr(model, 'jacobian_mul', counted)
    width = builder._model_dof()*builder._model_order()
    for _ in range(width + 2):
        np.testing.assert_allclose(runtime.residual_jvp(v), J @ v, rtol=2e-10, atol=1e-8)
    before = len(calls)
    actual = runtime.residual_jvp(v)
    np.testing.assert_allclose(actual, J @ v, rtol=2e-10, atol=1e-8)
    np.testing.assert_allclose(runtime.residual_vjp(w), J.T @ w, rtol=2e-10, atol=1e-8)
    assert sum(n > 1 for n in calls) == 1
    assert len(calls) == before + (0 if matrix_rhs else 1)
    assert bool(builder._motion_product_cache) == matrix_rhs
    runtime.pack.set(runtime.pack.get() + rng.normal(size=runtime.pack.n_total)*.01)
    actual = runtime.residual_jvp(v)
    assert not builder._motion_product_cache
    _, new_J = runtime.linearize()
    np.testing.assert_allclose(actual, new_J @ v, rtol=2e-10, atol=1e-8)
