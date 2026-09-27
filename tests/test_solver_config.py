"""TOML solver defaults survive compilation and runtime views."""
import copy

import numpy as np
import pytest

from rei import compile_nls_problem_spec, compile_nls_problem_spec_toml, solve
from rei.optimize.builder import compile_nls_problem
from rei.optimize.dsl import problem_spec_to_dsl
from rei.optimize.reductions import build_nullspace_equality_reduction
from rei.problem import as_linearized_problem


def spec():
    return {
        'variables': {'x': {'init': [0., 0.]}},
        'terms': [{'var': 'x', 'target': [1., 2.]}],
        'solver': {'name': 'gauss_newton_krylov',
                   'options': {'preconditioner': 'normal', 'verbose': False, 'max_iters': 20}},
    }


def compile_spec(value):
    return compile_nls_problem_spec(value, build_state=lambda *a, **kw: {})


def test_toml_solver_configuration_runs_without_python_solver_arguments(tmp_path):
    path = tmp_path / 'problem.toml'
    path.write_text('''
[solver]
name = "gauss_newton_krylov"
[solver.options]
preconditioner = "normal"
verbose = false
[variables.x]
init = [0.0, 0.0]
[[terms]]
var = "x"
target = [1.0, 2.0]
''')
    runtime = compile_nls_problem_spec_toml(path, build_state=lambda *a, **kw: {})
    out = solve(runtime)
    assert out.converged
    np.testing.assert_allclose(out.solution, [1., 2.], atol=1e-7)
    assert out.meta['inner_solves'][0]['preconditioner'] == 'normal'


def test_options_override_and_do_not_mutate_configuration():
    original = spec()
    snapshot = copy.deepcopy(original)
    runtime = compile_spec(original)
    out = solve(runtime, solver='gauss_newton_krylov', options={'max_iters': 0})
    assert out.iterations == 0
    assert original == snapshot
    assert runtime.solver_config == original['solver']
    assert solve(runtime).converged


def test_explicit_different_solver_does_not_inherit_incompatible_options():
    runtime = compile_spec(spec())
    # LM would reject the Krylov-only preconditioner option if it leaked.
    out = solve(runtime, solver='levenberg_marquardt', options={'verbose': False})
    assert out.converged


def test_raw_dsl_defaults_are_copied_and_used():
    dsl = problem_spec_to_dsl(spec())
    runtime = compile_nls_problem(dsl, build_state=lambda *a, **kw: {})
    dsl['solver']['options']['max_iters'] = 0
    assert solve(runtime).converged


@pytest.mark.parametrize('adapt', [False, True])
def test_reduced_runtime_inherits_configuration(adapt):
    value = spec()
    value['variables']['y'] = {'init': [3.]}
    value['terms'].append({'var': 'y', 'target': [0.], 'constraint': 'eq'})
    runtime = compile_spec(value)
    reduced = build_nullspace_equality_reduction(runtime).runtime
    assert reduced.solver_config == runtime.solver_config
    view = as_linearized_problem(reduced) if adapt else reduced
    assert view.solver_config == runtime.solver_config
    out = solve(view)
    assert out.converged
    assert out.meta["inner_solves"][0]["preconditioner"] == "normal"


@pytest.mark.parametrize('config', [
    'gauss_newton', {}, {'name': ''}, {'name': 42},
    {'name': 'gauss_newton', 'max_iters': 10},
    {'name': 'gauss_newton', 'options': []},
])
def test_invalid_config_is_rejected(config):
    value = spec()
    value['solver'] = config
    with pytest.raises(ValueError, match='solver'):
        compile_spec(value)


@pytest.mark.parametrize('config', [
    {'name': 'does_not_exist'},
    {'name': 'gauss_newton', 'options': {'preconditioner': 'normal'}},
])
def test_unknown_solver_and_incompatible_options_are_rejected(config):
    value = spec()
    value['solver'] = config
    with pytest.raises(ValueError):
        solve(compile_spec(value))
