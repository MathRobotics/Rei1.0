"""Research-only Krylov variants; never modifies production source files.

Run from repository root with the pinned RoboKots checkout on PYTHONPATH.
The experimental monkey patches are restored even on failure.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import cProfile
import importlib.util
import inspect
import json
from pathlib import Path
import pstats
import textwrap
import time
from types import SimpleNamespace

import numpy as np
from rei import solve
import rei.optimize.runtime as runtime_module
import rei.optimize.solvers.gauss_newton_krylov as krylov

spec = importlib.util.spec_from_file_location(
    'doc_benchmark', Path(__file__).with_name('kots_doc_operator.py'))
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@contextmanager
def single_term_fused():
    """Let the existing summed VJP route also handle one dynamics stack."""
    original = runtime_module.NLSRuntime._batched_dynamics_residual_vjp
    source = textwrap.dedent(inspect.getsource(original))
    old = 'len(candidates) < 2'
    if source.count(old) != 1:
        raise RuntimeError('Runtime changed; review the research patch before running.')
    namespace = {}
    exec(source.replace(old, 'len(candidates) < 1'), vars(runtime_module), namespace)
    runtime_module.NLSRuntime._batched_dynamics_residual_vjp = namespace[original.__name__]
    try:
        yield
    finally:
        runtime_module.NLSRuntime._batched_dynamics_residual_vjp = original


def sketch16(model, J, mode, *, probes, max_size, floor, rng):
    """Experimental rank-16 nonlinear curvature sketch in affine coordinates.

    This is deliberately a research prototype, not a supported preconditioner.
    It uses 16 JVP/VJP pairs per build and refreshes after five accepted steps.
    """
    gram = model.linear_residual_gram(max_size=max_size)
    d, q = np.linalg.eigh((gram + gram.T) / 2)
    d = np.maximum(d, floor * d.max())
    whitening = q / np.sqrt(d)
    omega = rng.standard_normal((len(d), 16))
    y = np.column_stack([
        whitening.T @ (J.T @ (J @ (whitening @ o)) - gram @ (whitening @ o))
        for o in omega.T])
    core = (omega.T @ y + y.T @ omega) / 2
    eigenvalues, vectors = np.linalg.eigh(core)
    keep = eigenvalues > 1e-10 * max(1., eigenvalues.max())
    factor = y @ (vectors[:, keep] / np.sqrt(eigenvalues[keep]))
    u, singular, _ = np.linalg.svd(factor, full_matrices=False)
    shrink = singular**2 / (1 + singular**2)

    def inverse(v):
        z = whitening.T @ v
        return whitening @ (z - u @ (shrink * (u.T @ z)))

    return inverse, 'experimental_sketch'


def settings(model='7_dof_arm.urdf', torque_weight=None):
    return SimpleNamespace(model=model, steps=201, controls=50,
        torque_weight=torque_weight, max_iters=30, tol_grad=1e-8)


def sweep():
    records = []
    original = krylov._preconditioner
    try:
        for name, options in [('baseline', {}), ('looser', {'forcing_min': .01}),
                              ('inner100', {'inner_max_iters': 100}), ('sketch16', {})]:
            _, reduction, _ = benchmark.make_problem(settings(torque_weight=1e-6))
            krylov._preconditioner = sketch16 if name == 'sketch16' else original
            start = time.perf_counter()
            out = solve(reduction.runtime, solver='gauss_newton_krylov',
                        options={'max_iters': 50, 'verbose': False, **options})
            elapsed = time.perf_counter() - start
            r, jac = reduction.runtime.linearize()
            record = dict(name=name, seconds=elapsed, status=out.status,
                objective=out.stats.objective, gradient=float(np.max(np.abs(jac.T @ r))),
                inner=sum(x['iterations'] for x in out.meta['inner_solves']),
                trials=out.iterations, options=options)
            records.append(record)
            print(record, flush=True)
    finally:
        krylov._preconditioner = original
    return records


def compare():
    records = []
    for model in ('planar2.urdf', '7_dof_arm.urdf'):
        args = settings(model)
        for repeat in range(4):
            for name in (('baseline', 'fused') if repeat % 2 == 0 else ('fused', 'baseline')):
                if name == 'fused':
                    with single_term_fused():
                        result = benchmark.run(args, 'gauss_newton_krylov')
                else:
                    result = benchmark.run(args, 'gauss_newton_krylov')
                result.update(variant=name, model=model, repeat=repeat)
                if repeat:  # First pair is warmup.
                    records.append(result)
                print(model, name, result['seconds'], result['gradient_inf'], flush=True)
    return records


def verify():
    rng = np.random.default_rng(2026)
    records = []
    with single_term_fused():
        for model in ('planar2.urdf', '7_dof_arm.urdf'):
            for weight in (1e-10, 1e-6):
                _, reduction, native = benchmark.make_problem(settings(model, weight))
                rt = reduction.runtime
                initial = rt.pack.get().copy()
                for _ in range(3):
                    rt.pack.set(initial + .02 * rng.standard_normal(initial.size))
                    r, jac = rt.linearize()
                    v, w = rng.standard_normal(jac.shape[1]), rng.standard_normal(jac.shape[0])
                    calls = []
                    original = native.jacobian_transpose_mul_many
                    def counted(*args, **kwargs):
                        calls.append(1)
                        return original(*args, **kwargs)
                    native.jacobian_transpose_mul_many = counted
                    try:
                        jv, jtw = rt.residual_jvp(v), rt.residual_vjp(w)
                    finally:
                        native.jacobian_transpose_mul_many = original
                    np.testing.assert_allclose(jv, jac @ v, rtol=1e-9, atol=1e-9)
                    np.testing.assert_allclose(jtw, jac.T @ w, rtol=1e-9, atol=1e-9)
                    np.testing.assert_allclose(w @ jv, v @ jtw, rtol=1e-9, atol=1e-9)
                    assert calls, 'Fused native route was not used.'
                records.append(dict(model=model, torque_weight=weight, points=3, passed=True))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('compare', 'sweep', 'verify', 'profile'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'profile':
        _, reduction, _ = benchmark.make_problem(settings())
        profiler = cProfile.Profile()
        profiler.runcall(solve, reduction.runtime, solver='gauss_newton_krylov',
                        options={'verbose': False})
        profiler.dump_stats(str(args.output))
        pstats.Stats(profiler).strip_dirs().sort_stats('cumtime').print_stats(28)
    else:
        result = {'compare': compare, 'sweep': sweep, 'verify': verify}[args.mode]()
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
