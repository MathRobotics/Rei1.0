"""Reproduce the local NumPy / Pinocchio / RoboKots Rust timing comparison."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import robokots_native
from robokots.kots import Kots

from rei import load_problem_spec_toml, solve
from rei.optimize.reductions import build_nullspace_equality_reduction
from rei.optimize_backends.kots import compile_kots_trajectory_problem
from rei.optimize_backends.pinocchio import compile_pinocchio_trajectory_problem

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("robokots-numpy", "pinocchio", "robokots-rust")
GRAVITY = (0.0, 0.0, -9.81)


def compile_problem(backend, problem, path, *, jacobian_strategy="mul"):
    if backend == "pinocchio":
        model = pin.buildModelFromUrdf(str(path))
        model.gravity.linear = np.asarray(GRAVITY)
        return compile_pinocchio_trajectory_problem(problem, model=model, data=model.createData())
    engine = "rust" if backend == "robokots-rust" else "numpy"
    model = Kots.from_urdf_file(str(path), order=3, backend=engine)
    return compile_kots_trajectory_problem(
        problem, model=model, kots_backend=engine, gravity=GRAVITY,
        jacobian_method="analytic", jacobian_strategy=jacobian_strategy, batch_trajectory=True,
    )


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, 1000 * (time.perf_counter() - start)


def statistics_ms(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "samples": values}


def verify(problem, path, *, jacobian_strategy="mul"):
    # Unit torque weight prevents the optimization's small torque weight from
    # concealing cross-backend errors in dynamics and derivatives.
    torque = copy.deepcopy(problem)
    torque["terms"] = [copy.deepcopy(problem["terms"][-1])]
    torque["terms"][0]["cost"] = {"type": "scalar_weight", "w": 1.0}
    compiled = {b: compile_problem(b, torque, path, jacobian_strategy=jacobian_strategy) for b in BACKENDS}
    rng = np.random.default_rng(42)
    errors = {b: {"residual_max_abs": 0., "jacobian_max_abs": 0.} for b in BACKENDS if b != "pinocchio"}
    for _ in range(3):
        x = rng.normal(scale=.05, size=compiled["pinocchio"].runtime.pack.n_total)
        values = {}
        for backend, c in compiled.items():
            c.runtime.pack.set(x.copy())
            values[backend] = tuple(np.asarray(v).copy() for v in c.runtime.linearize())
        for backend, error in errors.items():
            for key, actual, expected in zip(error, values[backend], values["pinocchio"], strict=True):
                error[key] = max(error[key], float(np.max(np.abs(actual - expected))))
                np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
    native = compiled["robokots-rust"].state_builder.model
    return {"against_pinocchio": errors, "rust_state_type": type(native.outward_state_).__name__}


def measure(problem, path, repeat, warmup, solve_repeat, *, jacobian_strategy="mul"):
    rows = {}
    compiled = {}
    # First evaluation includes initialization; imports are already complete.
    for backend in BACKENDS:
        c, compile_ms = timed(lambda: compile_problem(backend, problem, path, jacobian_strategy=jacobian_strategy))
        (_, _), first_ms = timed(c.runtime.linearize)
        compiled[backend] = c
        rows[backend] = {"first_compile_ms": compile_ms, "first_linearize_ms": first_ms,
                         "compile_ms": [], "linearize_ms": [], "reduction_ms": [],
                         "solve_ms": [], "solves": []}
    n_total = compiled["pinocchio"].runtime.pack.n_total
    rng = np.random.default_rng(0)
    # Rotate backend ordering to reduce systematic order/temperature bias.
    for i in range(repeat + warmup):
        x = rng.normal(scale=.05, size=n_total)
        ordering = BACKENDS[i % 3:] + BACKENDS[:i % 3]
        for backend in ordering:
            _, compile_ms = timed(lambda: compile_problem(backend, problem, path, jacobian_strategy=jacobian_strategy))
            runtime = compiled[backend].runtime
            runtime.pack.set(x.copy())
            (r, J), linearize_ms = timed(runtime.linearize)
            if i >= warmup:
                rows[backend]["compile_ms"].append(compile_ms)
                rows[backend]["linearize_ms"].append(linearize_ms)
            rows[backend]["jacobian_shape"] = list(J.shape)
    for i in range(solve_repeat):
        ordering = BACKENDS[i % 3:] + BACKENDS[:i % 3]
        for backend in ordering:
            c = compile_problem(backend, problem, path, jacobian_strategy=jacobian_strategy)
            reduction, reduction_ms = timed(lambda: build_nullspace_equality_reduction(
                c.runtime, eq_selector_attr="enforce", eq_selector_value="nullspace"))
            outcome, solve_ms = timed(lambda: solve(
                reduction.runtime, solver="levenberg_marquardt",
                options={"max_iters": 500, "tol_dx": 1e-12, "verbose": False}))
            assert outcome.converged, (backend, outcome.stats)
            rows[backend]["reduction_ms"].append(reduction_ms)
            rows[backend]["solve_ms"].append(solve_ms)
            rows[backend]["solves"].append({"status": outcome.stats.status,
                "iterations": int(outcome.stats.iterations), "cost": float(outcome.stats.objective)})
            print(path.stem, backend, "solve", i + 1, f"{solve_ms:.1f} ms", flush=True)
    reference_cost = rows["pinocchio"]["solves"][0]["cost"]
    for row in rows.values():
        np.testing.assert_allclose([s["cost"] for s in row["solves"]], reference_cost, rtol=1e-7, atol=1e-9)
        for key in ("compile_ms", "linearize_ms", "reduction_ms", "solve_ms"):
            row[key] = statistics_ms(row[key])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--solve-repeat", type=int, default=3)
    parser.add_argument("--jacobian-strategy", choices=("dense", "mul"), default="dense",
                        help="Dense assembly suits this benchmark's dense LM solver; mul measures wide JVP assembly.")
    parser.add_argument("--output", type=Path, default=Path("/tmp/rei-three-backends.json"))
    args = parser.parse_args()
    assert args.repeat > 0 and args.warmup >= 0 and args.solve_repeat > 0
    result = {"platform": platform.platform(), "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in (
            "numpy", "scipy", "pin", "mathrobo", "robokots", "robokots-native")},
        "native_api": robokots_native.__api_version__,
        "robokots_source": json.loads(importlib.metadata.distribution("robokots").read_text("direct_url.json")),
        "thread_environment": {k: os.environ.get(k) for k in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "settings": vars(args) | {"output": str(args.output)}, "models": {}}
    problem = load_problem_spec_toml(ROOT / "examples/spec/pinocchio_traj_dynamics.toml")
    for name in ("planar2", "fr3v2.1_franka_hand"):
        path = ROOT / "examples/models" / f"{name}.urdf"
        print("Checking", name, flush=True)
        correctness = verify(problem, path, jacobian_strategy=args.jacobian_strategy)
        print("Measuring", name, correctness, flush=True)
        rows = measure(problem, path, args.repeat, args.warmup, args.solve_repeat,
                       jacobian_strategy=args.jacobian_strategy)
        result["models"][name] = {"correctness": correctness, "timings": rows}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        for backend, row in rows.items():
            print(name, backend, {k: row[k]["median"] for k in (
                "compile_ms", "linearize_ms", "solve_ms")}, flush=True)


if __name__ == "__main__":
    main()
