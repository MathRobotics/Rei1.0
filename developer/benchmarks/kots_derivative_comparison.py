"""Compare fixed-trajectory torque Jacobians, then identically configured DOC/IOC.

PYTHONPATH=. .venv/bin/python developer/benchmarks/kots_derivative_comparison.py \
    --steps 21 --control-points 8 --output /tmp/kots-comparison.json

The report contains effective DSL, initial/fixed points, settings, timings,
evaluation counts, solver outcomes, full KKT checks and IOC results. No failed
configuration or unsupported output is silently omitted.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from robokots.kots import Kots

from rei import load_problem_spec_toml, solve
from rei.optimize.kkt import check_kkt_conditions
from rei.optimize.reductions import build_nullspace_equality_reduction
from rei.optimize_backends.kots import compile_kots_trajectory_problem
from rei.optimize_backends.trajectory_ioc import estimate_ioc_weights


ROOT = Path(__file__).resolve().parents[2]


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def instrument(obj, names, counts):
    for name in names:
        original = getattr(obj, name)
        def counted(*args, _fn=original, _name=name, **kwargs):
            counts[_name] = counts.get(_name, 0) + 1
            return _fn(*args, **kwargs)
        setattr(obj, name, counted)


def compare(args):
    spec = load_problem_spec_toml(args.spec)
    if args.steps is not None:
        duration = spec["time"]["N"] * spec["time"]["dt"]
        spec["time"].update(N=args.steps - 1, dt=duration / (args.steps - 1))
    if args.control_points is not None:
        spec["trajectory"]["num_ctrl_points"] = args.control_points
    options = dict(max_iters=args.max_iters, tol_grad=args.tol_grad,
                   tol_r=args.tol_r, tol_dx=args.tol_dx, verbose=False)
    report = {
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "jax": importlib.metadata.version("jax"),
            "robokots": importlib.metadata.distribution("robokots").read_text("direct_url.json"),
        },
        "spec_path": str(args.spec),
        "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "effective_dsl": spec, "seed": args.seed, "gravity": args.gravity,
        "solver": args.solver, "solver_options": options, "warm_repeats": args.repeats,
        "timing_note": "first includes compile+execution; warm median uses the identical fixed point, "
                       "with Rei state reuse disabled. DOC follows fixed-point warmup, after restoring "
                       "the TOML initial point; any additional shape compilation is included in DOC.",
        "results": [],
    }
    reference = None
    fixed = None
    for method in ("analytic", "numerical", "autodiff"):
        for batch in (False, True):
            derivative_options = ({"eps": args.eps} if method == "numerical" else
                                  {"mode": args.ad_mode, "jit": args.jit} if method == "autodiff" else {})
            compiled = compile_kots_trajectory_problem(
                spec, model=Kots.from_urdf_file(str(args.model), backend=args.backend),
                kots_backend=args.backend, gravity=args.gravity, jacobian_method=method,
                jacobian_options=derivative_options, batch_trajectory=batch,
                jacobian_strategy=args.strategy, max_derivative_order=args.model_order - 1,
            )
            runtime, builder = compiled.runtime, compiled.state_builder
            initial = runtime.pack.get().copy()
            if fixed is None:
                fixed = np.random.default_rng(args.seed).normal(scale=.1, size=len(initial))
                report.update(initial_point=initial, fixed_point=fixed)
            np.testing.assert_array_equal(initial, report["initial_point"])
            keys = sorted((key for key in runtime.required_list()
                           if key.dtype == "dynamics" and key.field == "torque_J_p"), key=lambda key: key.k)
            if not keys:
                raise ValueError("The comparison TOML must request total-joint torque derivatives w.r.t. p.")
            counts = {}
            instrument(builder.model, ("jacobian", "jacobian_autodiff", "jacobian_mul",
                                       "jacobian_transpose_mul", "jacobian_transpose_mul_many", "dynamics"), counts)
            instrument(runtime, ("linearize_stacked_terms", "eval_stacked_terms"), counts)
            times = []
            for _ in range(args.repeats + 1):
                runtime.pack.set(fixed)
                runtime.state.invalidate()
                builder._batched_dynamics_cache_key = None
                start = perf_counter()
                values = builder.build_state(fixed, required=keys, pack=runtime.pack, time=runtime.time)
                matrix = np.concatenate([values[key] for key in keys], axis=0)
                times.append(perf_counter() - start)
            if reference is None:
                reference = matrix.copy()
            difference = matrix - reference
            row = {
                "settings": compiled.derivative_settings,
                "model_order": compiled.model_order,
                "fixed_jacobian": {
                    "shape": matrix.shape, "first_seconds": times[0],
                    "warm_median_seconds": float(np.median(times[1:])),
                    "warm_seconds": times[1:], "max_abs_error": float(np.max(np.abs(difference))),
                    "relative_frobenius_error": float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-30)),
                    "counts": dict(counts),
                },
            }
            runtime.pack.set(initial)
            runtime.state.invalidate()
            counts.clear()
            start = perf_counter()
            reduction = build_nullspace_equality_reduction(runtime, eq_selector_attr="enforce", eq_selector_value="nullspace")
            row["reduction_setup_seconds"] = perf_counter() - start
            counts.clear()
            start = perf_counter()
            outcome = solve(reduction.runtime, solver=args.solver, options=options)
            row["doc"] = dict(seconds=perf_counter() - start, status=outcome.status,
                              converged=outcome.converged, iterations=outcome.stats.iterations,
                              objective=outcome.stats.objective, counts=dict(counts))
            full_point = reduction.lift(outcome.solution)
            runtime.pack.set(full_point)
            runtime.state.invalidate()
            counts.clear()
            start = perf_counter()
            row["kkt"] = asdict(check_kkt_conditions(runtime))
            row["kkt_seconds"] = perf_counter() - start
            row["kkt_counts"] = dict(counts)
            counts.clear()
            start = perf_counter()
            row["ioc"] = estimate_ioc_weights(compiled, p=full_point)
            row["ioc_seconds"] = perf_counter() - start
            row["ioc_counts"] = dict(counts)
            row["solution"] = full_point
            report["results"].append(row)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "examples/spec/robokots_traj_dynamics_d12.toml")
    parser.add_argument("--model", type=Path, default=ROOT / "examples/models/planar2.urdf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--control-points", type=int)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--backend", choices=("numpy", "rust"), default="rust")
    parser.add_argument("--strategy", choices=("dense", "mul"), default="dense")
    parser.add_argument("--model-order", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--ad-mode", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gravity", type=float, nargs=3, default=(0., -9.81, 0.))
    parser.add_argument("--solver", default="gauss_newton")
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--tol-grad", type=float, default=1e-8)
    parser.add_argument("--tol-r", type=float, default=1e-10)
    parser.add_argument("--tol-dx", type=float, default=1e-12)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path.")
    if args.repeats < 1 or (args.steps is not None and args.steps < 2):
        parser.error("repeats >= 1 and steps >= 2 are required")
    report = compare(args)
    with args.output.open("x") as stream:
        json.dump(report, stream, default=json_default, indent=2, allow_nan=False)
    for row in report["results"]:
        settings, fixed, doc = row["settings"], row["fixed_jacobian"], row["doc"]
        print(f"{settings['jacobian_method']:9} batch={settings['batch_trajectory']} "
              f"first={fixed['first_seconds']:.4f}s warm={fixed['warm_median_seconds']:.4f}s "
              f"error={fixed['max_abs_error']:.3g} DOC={doc['status']} "
              f"iterations={doc['iterations']} time={doc['seconds']:.4f}s")


if __name__ == "__main__":
    main()
