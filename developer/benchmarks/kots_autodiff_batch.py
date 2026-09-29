"""Measure cold JIT and warm trajectory linearization separately.

Run: PYTHONPATH=. .venv/bin/python developer/benchmarks/kots_autodiff_batch.py
Every timed linearization uses a new point and invalidates Rei's state cache.
The first measurement includes JIT compilation; warm times are medians.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from robokots.kots import Kots

from rei import load_problem_spec_toml
from rei.optimize_backends.kots import compile_kots_trajectory_problem


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--backend", choices=("numpy", "rust"), default="rust")
    args = parser.parse_args()
    results = []
    for method, batch in (("analytic", True), ("autodiff", False), ("autodiff", True)):
        spec = load_problem_spec_toml(ROOT / "examples/spec/robokots_traj_dynamics_d12.toml")
        spec["time"].update(N=args.steps - 1, dt=2 / (args.steps - 1))
        spec["trajectory"]["num_ctrl_points"] = 8
        model = Kots.from_urdf_file(str(ROOT / "examples/models/planar2.urdf"), backend=args.backend)
        compiled = compile_kots_trajectory_problem(
            spec, model=model, jacobian_method=method, batch_trajectory=batch,
            kots_backend=args.backend, gravity=(0., -9.81, 0.), max_derivative_order=4,
        )
        runtime = compiled.runtime
        point = np.random.default_rng(18).normal(scale=.1, size=runtime.pack.n_total)
        durations = []
        for index in range(args.repeats + 1):
            runtime.pack.set(point + index * 1e-4)
            runtime.state.invalidate()
            start = perf_counter()
            runtime.linearize()
            durations.append(perf_counter() - start)
        results.append(dict(method=method, batch=batch, first_seconds=durations[0],
                            warm_median_seconds=float(np.median(durations[1:]))))
    print(json.dumps(dict(steps=args.steps, dof=2, model_order=5, repeats=args.repeats,
                          backend=args.backend, jax=jax.__version__, devices=str(jax.devices()),
                          results=results), indent=2))


if __name__ == "__main__":
    main()
