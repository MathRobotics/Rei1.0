"""Trust-region GN correctness, globalization, bounded products and preconditioning."""
import json
import numpy as np
import pytest

from rei import solve, solve_gauss_newton_krylov
from rei.optimize.solvers.gauss_newton_krylov import _boundary, _truncated_cg
from rei.optimize.solvers._jacobian_operator import JacobianProducts
from rei.core.expr.nodes import GetVarExpr, SubExpr, StackExpr, HingeExpr
from rei.core.expr.types import Variable, VariablePack, RuntimeContext
from rei.optimize.costs import L2Cost, ScalarWeightCost, HuberCost
from rei.optimize.runtime import NLSRuntime
from rei.problem import NLSProblem, as_linearized_problem


class DiagonalProblem:
    def __init__(self, n=3):
        self.n_total = n
        self.x = np.zeros(n)
        self.scales = np.linspace(1., 10., n)
        self.calls = [0, 0]
    def get_point(self): return self.x.copy()
    def set_point(self, x): self.x = np.asarray(x, dtype=float).copy()
    def required_list(self, required=None): return [] if required is None else list(required)
    def eval(self, *, required=None): return self.scales*(self.x-1.)
    def jvp(self, v, *, required=None):
        self.calls[0] += 1
        return self.scales*v
    def vjp(self, w, *, required=None):
        self.calls[1] += 1
        return self.scales*w
    def linearize(self, **kwargs): raise AssertionError('nonlinear dense Jacobian forbidden')


class Nonlinear(DiagonalProblem):
    def __init__(self, x=.1):
        super().__init__(1)
        self.x[:] = x
    def eval(self, *, required=None): return self.x**2-1.
    def jvp(self, v, *, required=None): return 2*self.x*v
    def vjp(self, w, *, required=None): return 2*self.x*w


def run(model, **options):
    return solve(model, solver='gauss_newton_krylov',
                 options={'verbose': False, 'globalization': 'trust_region', **options})


@pytest.mark.parametrize('n', [1, 4, 10])
def test_pcg_interior_step_matches_independent_linear_solution(n):
    model=DiagonalProblem(n)
    J=JacobianProducts(model, [], n)
    g=model.vjp(model.eval())
    h, Jh, info=_truncated_cg(J,g,1e3,lambda v:v/model.scales**2,
                             tolerance=1e-10,max_iters=20)
    np.testing.assert_allclose(h,np.ones(n),atol=1e-10)
    np.testing.assert_allclose(Jh,model.scales,atol=1e-10)
    assert info['status']=='converged' and info['iterations']==1


@pytest.mark.parametrize('radius',[1e-5, .1, .9])
def test_boundary_step_lies_on_sphere_and_decreases_model(radius):
    model=DiagonalProblem(5)
    J=JacobianProducts(model, [], 5)
    g=model.vjp(model.eval())
    h,Jh,info=_truncated_cg(J,g,radius,lambda v:v,tolerance=1e-10,max_iters=10)
    assert info['status']=='boundary'
    assert np.linalg.norm(h)==pytest.approx(radius)
    assert 2*g@h+Jh@Jh < 0


def test_boundary_intersection_when_direction_points_inward():
    h=np.array([.8,.1]);p=np.array([-3.,1.])
    point=_boundary(h,p,1.)
    assert np.linalg.norm(point)==pytest.approx(1.)
    assert np.dot(point-h,p)>0


@pytest.mark.parametrize('mode',['identity','diagonal','auto'])
def test_matrix_free_solution_and_gradient(mode):
    model=DiagonalProblem(12)
    out=run(model,preconditioner=mode,tol_grad=1e-9)
    assert out.converged
    np.testing.assert_allclose(out.solution,1.,atol=1e-9)
    assert out.history[-1]['jt_r_inf_norm']<=1e-9


def test_product_budget_does_not_grow_with_number_of_columns():
    # With a diagonal map, every Rademacher probe gives exact diagonal curvature.
    model=DiagonalProblem(2000)
    out=run(model,max_iters=1,preconditioner_probes=4,inner_max_iters=2)
    assert out.converged
    assert model.calls[0] <= 4
    assert model.calls[1] <= 9


def test_nonlinear_rejection_history_and_state_restoration(tmp_path):
    model=Nonlinear()
    path=tmp_path/'states.jsonl'
    callbacks=[]
    out=solve_gauss_newton_krylov(model,history_vectors=True,history_path=path,verbose=False,
        globalization='trust_region', on_iter=lambda k,r,dx,g:callbacks.append((k,g.copy())))
    assert out.converged
    assert any(not e['accepted'] for e in out.trial_history)
    assert out.line_search_history==[]
    assert json.loads(path.read_text().splitlines()[-1])==out.history[-1]
    assert [json.loads(s) for s in (tmp_path/'states.trust_region_trials.jsonl').read_text().splitlines()]==out.trial_history
    for row in out.history:
        if 'variables' in row:
            x=row['variables'][0]
            assert row['objective']==pytest.approx((x*x-1)**2)
            assert row['jt_r']==pytest.approx([2*x*(x*x-1)])
    assert len(callbacks)==out.iterations


def test_evaluation_exception_restores_base_point_and_writes_trial(tmp_path):
    model=Nonlinear()
    def residual(*,required=None):
        if model.x[0]>1: raise ValueError('outside domain')
        return model.x**2-1
    model.eval=residual
    path=tmp_path/'states.jsonl'
    with pytest.raises(ValueError,match='outside domain'):
        run(model,history_path=path)
    np.testing.assert_array_equal(model.x,[.1])
    rows=[json.loads(s) for s in (tmp_path/'states.trust_region_trials.jsonl').read_text().splitlines()]
    assert rows[-1]['reason']=='evaluation_error'


def test_nonfinite_trials_are_rejected_and_recover():
    model=Nonlinear()
    model.eval=lambda **kw: np.array([np.inf]) if model.x[0]>2 else model.x**2-1
    out=run(model)
    assert out.converged
    assert any(row['objective'] is None and not row['accepted'] for row in out.trial_history)


@pytest.mark.parametrize('options',[{'inner_max_iters':0},{'max_iters':1.5},
    {'forcing_min':0},{'forcing_max':1},{'initial_radius':-1},{'max_radius':0},
    {'preconditioner_probes':0},{'preconditioner':'bad'},{'preconditioner_floor':0},
    {'preconditioner_max_size':-1},{'seed':-1},{'acceptance':.25},{'tol_grad':np.nan}])
def test_invalid_options_checked_before_x0_change(options):
    model=Nonlinear(1.)
    with pytest.raises(ValueError):
        solve(model,solver='gauss_newton_krylov',x0=[10.],options=options)
    np.testing.assert_array_equal(model.x,[1.])


@pytest.mark.parametrize('n',[0,3])
def test_initial_stationarity_does_not_build_preconditioner(n):
    model=DiagonalProblem(n);model.x[:]=1
    model.linear_residual_gram=lambda **kw: (_ for _ in ()).throw(AssertionError('unneeded preconditioner'))
    out=run(model)
    assert out.converged and out.iterations==0


def test_zero_iterations_and_stream_only_history(tmp_path):
    path=tmp_path/'states.jsonl'
    out=run(Nonlinear(),max_iters=0,history=False,history_path=path)
    assert out.status=='max_iters'
    assert out.history==out.trial_history==[]
    assert json.loads(path.read_text().splitlines()[-1])['status']=='max_iters'


def test_affine_gram_respects_selection_variable_order_and_ignores_nonlinear(monkeypatch):
    x=Variable('x',np.array([2.,-1.]));y=Variable('y',np.array([3.,4.]))
    pack=VariablePack([y,x])
    a,b=GetVarExpr('x',[x]),GetVarExpr('y',[y])
    expr=StackExpr('stack',[SubExpr('diff',a,b),a])
    hinge=HingeExpr('hinge',a)
    runtime=NLSRuntime(NLSProblem(pack,[(expr,ScalarWeightCost(3.)),(b,L2Cost()),
                                       (hinge,L2Cost()),(expr,HuberCost(.5))]),RuntimeContext(pack))
    view=as_linearized_problem(runtime,term_indices=[0,2,3])
    _,J=runtime.linearize_stacked_terms(term_indices=[0])
    monkeypatch.setattr(HingeExpr,'eval',lambda *a: (_ for _ in ()).throw(AssertionError('nonlinear preconditioner')))
    np.testing.assert_allclose(view.linear_residual_gram(),J.T@J)
    assert view.linear_residual_gram(max_size=3) is None


def test_nonzero_residual_stationary_solution():
    model=DiagonalProblem(1)
    model.eval=lambda **kw: np.array([model.x[0]-1,1.])
    model.jvp=lambda v,**kw: np.array([v[0],0.])
    model.vjp=lambda w,**kw: w[:1]
    out=run(model)
    assert out.converged
    assert out.stats.objective==pytest.approx(1.)
    assert out.solution==pytest.approx([1.])


def test_rosenbrock_converges_with_rejections_and_no_dense_derivatives():
    model=DiagonalProblem(2)
    model.x=np.array([-1.2,1.])
    model.eval=lambda **kw:np.array([10*(model.x[1]-model.x[0]**2),1-model.x[0]])
    model.jvp=lambda v,**kw:np.array([10*v[1]-20*model.x[0]*v[0],-v[0]])
    model.vjp=lambda w,**kw:np.array([-20*model.x[0]*w[0]-w[1],10*w[0]])
    out=run(model,tol_grad=1e-9)
    assert out.converged
    np.testing.assert_allclose(out.solution,[1.,1.],atol=1e-8)
    assert any(not e['accepted'] for e in out.trial_history)


def test_roundoff_acceptance_checks_gradient_progress():
    model=DiagonalProblem(1)
    model.eval=lambda **kw:np.array([1.,1e-12*(model.x[0]+1)])
    model.jvp=lambda v,**kw:np.array([0.,1e-12*v[0]])
    model.vjp=lambda w,**kw:np.array([1e-12*w[1]])
    out=run(model,tol_grad=1e-25)
    assert out.converged
    assert out.solution==pytest.approx([-1.],abs=1e-9)
    assert any(row.get('reason')=='gradient_progress' for row in out.trial_history)


def test_trust_history_formatter_exposes_radius_and_trials():
    from rei.optimize.history import format_solver_history
    out=run(Nonlinear())
    assert 'radius' in format_solver_history(out.history)
    assert 'trial 1' in format_solver_history(out.trial_history)
    assert 'trial 1' not in format_solver_history(out.trial_history,include_line_search=False)


def test_preconditioner_size_cap_avoids_provider_allocation():
    model=DiagonalProblem(20)
    model.linear_residual_gram=lambda **kw: (_ for _ in ()).throw(AssertionError('size cap ignored'))
    assert run(model,preconditioner_max_size=5).converged


def test_default_line_search_matches_dense_gauss_newton():
    model = Nonlinear()
    dense = Nonlinear()
    dense.linearize = lambda **kw: (dense.eval(), np.diag(2*dense.x))
    baseline = solve(dense, solver='gauss_newton', options={'verbose': False})
    out = solve(model, solver='gauss_newton_krylov', options={'verbose': False})
    assert out.converged and out.iterations == baseline.iterations
    np.testing.assert_allclose(out.solution, baseline.solution, atol=1e-10)
    assert out.line_search_history and not out.trial_history
    assert out.meta['solver'] == 'gauss_newton_krylov'


def test_default_line_search_has_bounded_preconditioner_products():
    model = DiagonalProblem(2000)
    out = solve(model, solver='gauss_newton_krylov', options={
        'verbose': False, 'max_iters': 1, 'preconditioner_probes': 4,
        'inner_max_iters': 2,
    })
    assert out.stats.objective < 1e-12 * out.stats.initial_objective
    assert model.calls[0] < 10 and model.calls[1] < 15
