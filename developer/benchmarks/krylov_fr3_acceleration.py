"""Compare shared torque JVPs and explicit normal preconditioning on FR3.

Requires the external doc_ioc.py loader and its native RoboKots environment.
Only forward optimization is run; no external project outputs are written.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from functools import wraps
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import rei
from rei.optimize.runtime import NLSRuntime
from rei.backends.state.robotics.kots import KotsTrajectoryStateBuilder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doc-script', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-iters', type=int, default=60)
    parser.add_argument('--disable-motion-cache', action='store_true')
    parser.add_argument('--variants', nargs='+', choices=('before', 'shared', 'normal', 'dense', 'uncached'),
                        default=['before', 'shared', 'normal'])
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('fr3_doc_input', args.doc_script)
    doc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(doc)
    with patch.object(sys, 'argv', [str(args.doc_script), '--random-weights', '--seed', '0']):
        config = doc.parse_args()
    dsl, steps, dt, controls, order = doc.load_dsl(config)
    doc.randomize_objective_weights(dsl, config.seed)
    model = doc.load_franka(config.model, order=order)
    compiled = doc.compile_trajectory_ioc_problem(
        dsl, backend='kots', model=model, data=None, gravity=(0., 0., -9.81),
        kots_backend=doc.KOTS_BACKEND, jacobian_method=config.jacobian_method)
    reduction = doc.build_nullspace_equality_reduction(compiled.runtime)
    initial = reduction.runtime.pack.get().copy()
    report = dict(python=platform.python_version(), numpy=np.__version__,
                  spec_sha256=hashlib.sha256(config.spec.read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(config.model.read_bytes()).hexdigest(),
                  settings=dict(max_iters=args.max_iters, tol_grad=1e-8, seed=0,
                                random_weights=True, verbose=False,
                                disable_motion_cache=args.disable_motion_cache),
                  dimensions=dict(steps=steps, dt=dt, controls=controls, order=order,
                                  variables=initial.size), initial=initial.tolist(), runs=[])
    for mode in args.variants:
        counts = {}
        options = dict(max_iters=args.max_iters, tol_grad=1e-8, verbose=False)
        if mode == 'normal':
            options['preconditioner'] = 'normal'
        with ExitStack() as stack:
            if args.disable_motion_cache or mode in ('uncached', 'before'):
                stack.enter_context(patch.object(
                    KotsTrajectoryStateBuilder, '_torque_motion_product',
                    lambda self, refs, direction, keys: self.model.jacobian_mul(refs, direction)))
            if mode == 'before':
                stack.enter_context(patch.object(NLSRuntime, '_shared_dynamics_jvp',
                                                 lambda self, *a: {}))
            for name in ('jacobian_mul', 'jacobian_transpose_mul_many'):
                original = getattr(model, name)
                @wraps(original)
                def counted(*a, _name=name, _fn=original, **kw):
                    counts[_name] = counts.get(_name, 0) + 1
                    return _fn(*a, **kw)
                stack.enter_context(patch.object(model, name, counted))
            start = perf_counter()
            def progress(k, residual_norm, step_norm, gradient):
                print(mode, k, round(perf_counter()-start, 2),
                      float(np.max(np.abs(gradient), initial=0.)), flush=True)
            out = rei.solve(reduction.runtime,
                            solver='gauss_newton' if mode == 'dense' else 'gauss_newton_krylov',
                            x0=initial, options=options, on_iter=progress)
            elapsed = perf_counter()-start
        # Dense validation is deliberately outside the timer and API counters.
        residual, jacobian = reduction.runtime.linearize()
        row = dict(mode=mode, seconds=elapsed, status=out.status, iterations=out.iterations,
                   objective=out.stats.objective,
                   gradient=float(np.max(np.abs(jacobian.T @ residual), initial=0.)),
                   counts=counts, inner=out.meta.get('inner_solves', []),
                   solution=out.solution.tolist(), history=out.history)
        report['runs'].append(row)
        args.output.write_text(json.dumps(report, indent=2))
        print(mode, elapsed, out.status, out.stats.objective, row['gradient'], flush=True)


if __name__ == '__main__':
    main()
