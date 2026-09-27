"""Profile FR3 forward solves and isolate the single-stack fused VJP change.

Uses the external doc_ioc.py input loader; never runs IOC or writes its output.
Timers overlap: adapter totals include native calls and must not be added to them.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
from functools import wraps
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import sys
import textwrap
from time import perf_counter
import numpy as np
import rei
from rei.optimize.runtime import NLSRuntime
from rei.problem.adapters import NLSRuntimeLinearProblem
import importlib
krylov = importlib.import_module('rei.optimize.solvers.gauss_newton_krylov')


@contextmanager
def timed_method(owner, name, label, totals):
    original = getattr(owner, name, None)
    if not callable(original):
        yield
        return
    @wraps(original)
    def wrapped(*args, **kwargs):
        start = perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            row = totals.setdefault(label, {'calls': 0, 'seconds': 0.})
            row['calls'] += 1
            row['seconds'] += perf_counter() - start
    setattr(owner, name, wrapped)
    try:
        yield
    finally:
        setattr(owner, name, original)


def main():
    from contextlib import ExitStack
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doc-script', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-iters', type=int, default=60)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('fr3_doc_input', args.doc_script)
    doc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(doc)
    original_argv = sys.argv
    sys.argv = [str(args.doc_script), '--random-weights', '--seed', '0']
    try:
        config = doc.parse_args()
    finally:
        sys.argv = original_argv
    current = NLSRuntime._batched_dynamics_residual_vjp
    source = textwrap.dedent(inspect.getsource(current))
    marker = 'minimum_candidates = 1 if callable(fused_vjp) else 2'
    if source.count(marker) != 1:
        raise RuntimeError('Review baseline patch: runtime has changed.')
    namespace = {}
    exec(source.replace(marker, 'minimum_candidates = 2'), current.__globals__, namespace)
    previous = namespace[current.__name__]
    report = {'python': platform.python_version(), 'numpy': np.__version__,
              'rei': rei.__file__, 'doc_script': str(args.doc_script.resolve()),
              'spec_sha256': hashlib.sha256(config.spec.read_bytes()).hexdigest(),
              'model_sha256': hashlib.sha256(config.model.read_bytes()).hexdigest(),
              'settings': {'max_iters': args.max_iters, 'tol_grad': 1e-8,
                           'random_weights': True, 'seed': 0,
                           'solver': 'gauss_newton_krylov', 'verbose': False},
              'timer_note': 'Adapter and native timers overlap; do not sum them.', 'runs': []}
    for repeat in range(args.repeats):
        for variant in (('before', 'after') if repeat % 2 == 0 else ('after', 'before')):
            dsl, steps, dt, controls, order = doc.load_dsl(config)
            doc.randomize_objective_weights(dsl, config.seed)
            model = doc.load_franka(config.model, order=order)
            compiled = doc.compile_trajectory_ioc_problem(
                dsl, backend='kots', model=model, data=None, gravity=(0., 0., -9.81),
                kots_backend=doc.KOTS_BACKEND, jacobian_method=config.jacobian_method)
            reduction = doc.build_nullspace_equality_reduction(compiled.runtime)
            initial = reduction.runtime.pack.get().copy()
            totals = {}
            print(f'START {variant} repeat={repeat} variables={initial.size}', flush=True)
            original_step = krylov.cgls_step
            def timed_step(*a, **kw):
                inverse = kw.get('apply_inverse')
                if inverse is not None:
                    def counted(v):
                        t = perf_counter()
                        try:
                            return inverse(v)
                        finally:
                            row = totals.setdefault('preconditioner.apply', {'calls': 0, 'seconds': 0.})
                            row['calls'] += 1
                            row['seconds'] += perf_counter() - t
                    kw['apply_inverse'] = counted
                return original_step(*a, **kw)
            NLSRuntime._batched_dynamics_residual_vjp = previous if variant == 'before' else current
            krylov.cgls_step = timed_step
            try:
                with ExitStack() as stack:
                    for name in ('eval', 'jvp', 'vjp', 'linear_residual_gram'):
                        stack.enter_context(timed_method(NLSRuntimeLinearProblem, name, 'adapter.'+name, totals))
                    stack.enter_context(timed_method(np.linalg, 'eigh', 'preconditioner.eigh', totals))
                    for name in ('jacobian_mul', 'jacobian_transpose_mul', 'jacobian_transpose_mul_many', 'dynamics', 'import_motions'):
                        stack.enter_context(timed_method(model, name, 'native.'+name, totals))
                    start = perf_counter()
                    def progress(k, residual_norm, step_norm, gradient):
                        print(f'ITER {variant} k={k} elapsed={perf_counter()-start:.2f} '
                              f'gradient={np.max(np.abs(gradient), initial=0.):.4g}', flush=True)
                    out = rei.solve(reduction.runtime, solver='gauss_newton_krylov', on_iter=progress,
                                    options={'max_iters': args.max_iters, 'tol_grad': 1e-8,
                                             'verbose': False})
                    elapsed = perf_counter() - start
                    print(f'SOLVED {variant} seconds={elapsed:.3f} status={out.status}', flush=True)
            finally:
                NLSRuntime._batched_dynamics_residual_vjp = current
                krylov.cgls_step = original_step
            # Independent dense validation is outside every measurement timer.
            residual, jacobian = reduction.runtime.linearize()
            row = dict(variant=variant, repeat=repeat, seconds=elapsed,
                       status=out.status,
                       reason=out.history[-1].get('reason', out.stats.message) if out.history else out.stats.message,
                       iterations=out.iterations,
                       objective=out.stats.objective,
                       gradient_inf=float(np.max(np.abs(jacobian.T @ residual), initial=0.)),
                       initial=initial.tolist(), solution=out.solution.tolist(),
                       inner_iterations=sum(x['iterations'] for x in out.meta['inner_solves']),
                       inner_limits=sum(x['status']=='max_iters' for x in out.meta['inner_solves']),
                       inner_solves=out.meta['inner_solves'], timers=totals,
                       solver_spans=[vars(s) for s in out.timing.spans],
                       history=out.history, dimensions=dict(steps=steps, dt=dt, controls=controls, order=order,
                                                          variables=initial.size, residuals=residual.size))
            report['runs'].append(row)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps({k:row[k] for k in ('variant','seconds','status','objective','gradient_inf','inner_iterations')}), flush=True)


if __name__ == '__main__':
    main()
