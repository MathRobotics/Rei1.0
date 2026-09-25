"""Compare DOC solvers from identical points using real RoboKots Rust calls."""
from __future__ import annotations
import argparse
import json
import time
import sys
import importlib.metadata
import platform
import os
import subprocess
import hashlib
import copy
from pathlib import Path
from collections import Counter
import numpy as np
from robokots.kots import Kots
from rei import load_problem_spec_toml, solve
from rei.optimize_backends.kots import compile_kots_trajectory_problem
from rei.optimize.reductions import build_nullspace_equality_reduction

ROOT = Path(__file__).resolve().parents[2]


def make_problem(args):
    spec = load_problem_spec_toml(ROOT/'examples/spec/robokots_traj_dynamics_d12.toml')
    spec['time']['N'] = args.steps - 1
    spec['time']['dt'] = 2. / (args.steps - 1)
    spec['trajectory']['num_ctrl_points'] = args.controls
    torque_weight = getattr(args, 'torque_weight', None)
    if torque_weight is not None:
        found = False
        for term in spec['terms']:
            if term['expr'].get('name') == 'torque_traj_regularization':
                term['cost']['w'] = torque_weight
                found = True
        if not found:
            raise ValueError('Torque term not found in normalized specification.')
    fields = getattr(args, 'torque_fields', ['torque'])
    if fields != ['torque']:
        index = next(i for i, term in enumerate(spec['terms'])
                     if term['expr'].get('name') == 'torque_traj_regularization')
        template = spec['terms'].pop(index)
        for field in fields:
            term = copy.deepcopy(template)
            term['expr']['name'] = field + '_traj_regularization'
            term['expr']['inner']['name'] = field + '_regularization_k'
            term['expr']['inner']['key']['field'] = field
            spec['terms'].append(term)
    order = max({'torque': 3, 'torque_d1': 4, 'torque_d2': 5}[field] for field in fields)
    model = Kots.from_urdf_file(str(ROOT/'examples/models'/args.model), order=order)
    compiled = compile_kots_trajectory_problem(spec, model=model, kots_backend='rust',
        jacobian_strategy='mul', gravity=(0., 0., -9.81))
    reduction = build_nullspace_equality_reduction(compiled.runtime,
        eq_selector_attr='enforce', eq_selector_value='nullspace')
    return compiled, reduction, model


def run(args, solver):
    compiled, reduction, model = make_problem(args)
    counts = Counter()
    seconds = Counter()
    for name in ('jacobian', 'jacobian_mul', 'jacobian_transpose_mul',
                 'jacobian_transpose_mul_many', 'import_motions', 'dynamics'):
        original = getattr(model, name, None)
        if not callable(original):
            continue
        def wrapped(*a, _fn=original, _name=name, **kw):
            key = _name
            if _name == 'jacobian_mul' and len(a) > 1:
                key += ':' + str(np.asarray(a[1]).shape)
            counts[key] += 1
            t = time.perf_counter()
            try:
                return _fn(*a, **kw)
            finally:
                seconds[_name] += time.perf_counter() - t
        setattr(model, name, wrapped)
    initial = reduction.runtime.pack.get().copy()
    options = dict(max_iters=args.max_iters, tol_grad=args.tol_grad,
                   verbose=False, history=True)
    if solver == 'gauss_newton_operator':
        options.update(inner_tol=args.inner_tol, inner_max_iters=args.inner_max_iters)
    t = time.perf_counter()
    out = solve(reduction.runtime, solver=solver, x0=initial, options=options)
    elapsed = time.perf_counter() - t
    # Independent dense check, outside timing and backend counters.
    before_counts, before_seconds = dict(counts), dict(seconds)
    r, J = reduction.runtime.linearize()
    gradient = float(np.linalg.norm(J.T @ r, ord=np.inf))
    feasibility = 0.
    eq = [i for i, attrs in enumerate(compiled.runtime.problem.term_attrs)
          if attrs.get('enforce') == 'nullspace']
    if eq:
        feasibility = np.linalg.norm(compiled.runtime.eval_stacked_terms(weighted=False, term_indices=eq))
    return dict(solver=solver, seconds=elapsed, status=out.status, iterations=out.iterations,
        objective=out.stats.objective, gradient_inf=gradient, feasibility=float(feasibility),
        variables=len(initial), residuals=len(r), calls=before_counts, backend_seconds=before_seconds,
        inner_iterations=sum(s['iterations'] for s in out.meta.get('inner_solves', [])),
        inner_limits=sum(s['status']=='max_iters' for s in out.meta.get('inner_solves', [])),
        preconditioner=out.meta.get('preconditioner'),
        solution=out.solution.tolist(),
        accepted=[dict(iteration=e['iteration'], objective=e['objective'], gradient=e['jt_r_inf_norm'])
                  for e in out.history if e['event'] in ('initial', 'iteration_end')])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='planar2.urdf')
    p.add_argument('--steps', type=int, default=41)
    p.add_argument('--controls', type=int, default=10)
    p.add_argument('--max-iters', type=int, default=30)
    p.add_argument('--tol-grad', type=float, default=1e-8)
    p.add_argument('--torque-weight', type=float, default=None)
    p.add_argument('--torque-fields', nargs='+', choices=['torque', 'torque_d1', 'torque_d2'],
                   default=['torque'], help='Dynamics residual fields; each uses the torque weight.')
    p.add_argument('--inner-tol', type=float, default=1e-8)
    p.add_argument('--inner-max-iters', type=int, default=None)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--solvers', nargs='+', choices=['gauss_newton', 'gauss_newton_operator', 'gauss_newton_krylov'],
                   default=['gauss_newton', 'gauss_newton_operator'])
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    if args.steps < 2 or args.controls < 6 or args.repeat < 1 or args.warmup < 0:
        p.error('steps >= 2, controls >= 6, repeat >= 1, warmup >= 0 are required')
    import robokots
    dist = importlib.metadata.distribution('robokots')
    source = dist.read_text('direct_url.json')
    import robokots._rust_core
    checkout = Path(robokots.__file__).resolve().parents[1]
    commit = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True) if (checkout/'.git').exists() else None
    rust_path = Path(robokots._rust_core.__file__)
    environment = dict(robokots_checkout_commit=commit.stdout.strip() if commit and commit.returncode == 0 else None,
        rust_extension=str(rust_path), rust_extension_sha256=hashlib.sha256(rust_path.read_bytes()).hexdigest(),
        python=sys.version, numpy=np.__version__, platform=platform.platform(),
        robokots=robokots.__file__, robokots_source=json.loads(source) if source else None,
        threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
    for _ in range(args.warmup):
        for solver in args.solvers:
            print(f'warmup: {solver}', flush=True)
            run(args, solver)
    records=[]
    for repeat in range(args.repeat):
        for solver in (args.solvers if repeat % 2 == 0 else list(reversed(args.solvers))):
            result=run(args,solver)
            records.append(result)
            print(f"{solver} {result['seconds']:.3f}s {result['status']} iters={result['iterations']} "
                  f"g={result['gradient_inf']:.3e} inner={result['inner_iterations']}", flush=True)
            args.output.write_text(json.dumps(dict(settings={k:str(v) if isinstance(v,Path) else v
                for k,v in vars(args).items()}, environment=environment, runs=records), indent=2))

if __name__=='__main__':
    main()
