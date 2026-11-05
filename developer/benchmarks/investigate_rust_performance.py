"""Isolate dense/JVP assembly and selected/RNEA dispatch in the same workload.

The specialized variants temporarily bypass selected-output dispatch for this
torque-only problem. They are diagnostic controls, not production API changes.
Run separately for each native wheel, with no concurrent build/benchmark.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import pstats
import statistics
import time
from pathlib import Path

import numpy as np
import robokots_native

from developer.benchmarks.compare_three_robotics_backends import (
    ROOT, build_nullspace_equality_reduction, compile_problem,
    load_problem_spec_toml, solve,
)

VARIANTS = ("pinocchio", "selected-mul", "selected-dense", "specialized-mul", "specialized-dense")


def compile_variant(variant, problem, path):
    c = compile_problem("pinocchio" if variant == "pinocchio" else "robokots-rust", problem, path)
    if variant != "pinocchio":
        c.state_builder.jacobian_strategy = variant.split("-")[1]
        if variant.startswith("specialized"):
            # All dynamics terms in this problem are ordinary joint torque.
            # Returning None selects the existing specialized torque fallback.
            c.state_builder.model._rust_selected_dynamics_apply = lambda *a, **kw: None
            c.state_builder.model._rust_selected_dynamics_jacobian = lambda *a, **kw: None
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--solve-repeat", type=int, default=3)
    parser.add_argument("--model", action="append", choices=("planar2", "fr3v2.1_franka_hand"))
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument("--solver", choices=("levenberg_marquardt", "gauss_newton"), default="levenberg_marquardt")
    parser.add_argument("--torque-step", type=int, default=1)
    args = parser.parse_args()
    assert args.repeat > 0 and args.solve_repeat > 0 and args.torque_step > 0
    native_dir = Path(robokots_native.__file__).parent
    result = {"label": args.label, "solver": args.solver, "torque_step": args.torque_step,
        "native_module": robokots_native.__file__,
        "native_hashes": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in native_dir.glob("*.so")}, "models": {}}
    problem = load_problem_spec_toml(ROOT / "examples/spec/pinocchio_traj_dynamics.toml")
    if args.torque_step > 1:
        problem["terms"][-1]["expr"]["range"] = {"at": list(range(0, problem["time"]["N"] + 1, args.torque_step))}
    for name in args.model or ("planar2", "fr3v2.1_franka_hand"):
        path = ROOT / "examples/models" / (name + ".urdf")
        rows = result["models"][name] = {}
        reference = None
        variants = args.variant or VARIANTS
        assert variants[0] == "pinocchio", "Pinocchio must run first as the correctness reference."
        for variant in variants:
            c = compile_variant(variant, problem, path)
            xs = np.random.default_rng(0).normal(scale=.05, size=(args.repeat + 2, c.runtime.pack.n_total))
            samples = []
            for i, x in enumerate(xs):
                c.runtime.pack.set(x.copy())
                start = time.perf_counter()
                r, J = c.runtime.linearize()
                if i >= 2:
                    samples.append(1000 * (time.perf_counter() - start))
            if reference is None:
                reference = (r.copy(), J.copy())
            np.testing.assert_allclose(r, reference[0], rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(J, reference[1], rtol=1e-8, atol=1e-8)
            # Also compare with unit torque weight (rescale only these rows),
            # since the optimization's 1e-10 cost weight scales residuals by 1e-5.
            torque_rows = len(range(0, c.trajectory_map.steps, args.torque_step)) * c.trajectory_map.q_dim
            np.testing.assert_allclose(r[-torque_rows:] * 1e5, reference[0][-torque_rows:] * 1e5,
                                       rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(J[-torque_rows:] * 1e5, reference[1][-torque_rows:] * 1e5,
                                       rtol=1e-8, atol=1e-8)
            # Profile separately from measured samples.
            c.runtime.pack.set(xs[0].copy())
            prof = cProfile.Profile()
            prof.runcall(c.runtime.linearize)
            stream = io.StringIO()
            pstats.Stats(prof, stream=stream).sort_stats("cumulative").print_stats(30)
            profile = stream.getvalue()
            products = []
            if variant != "pinocchio":
                model = c.state_builder.model
                original = model.jacobian_mul
                def watched(refs, rhs, **kwargs):
                    products.append(list(np.asarray(rhs).shape))
                    return original(refs, rhs, **kwargs)
                model.jacobian_mul = watched
                c.runtime.pack.set(xs[1].copy())
                c.runtime.linearize()
            solves = []
            for _ in range(args.solve_repeat):
                c = compile_variant(variant, problem, path)
                reduction = build_nullspace_equality_reduction(
                    c.runtime, eq_selector_attr="enforce", eq_selector_value="nullspace")
                start = time.perf_counter()
                out = solve(reduction.runtime, solver=args.solver,
                    options={"max_iters": 500, "tol_dx": 1e-12 if args.solver == "levenberg_marquardt" else 1e-8,
                             "verbose": False})
                solves.append({"ms": 1000 * (time.perf_counter()-start), "cost": out.stats.objective,
                    "iterations": out.stats.iterations, "converged": out.converged})
                assert out.converged
            row = {"linearize_ms": samples, "linearize_median_ms": statistics.median(samples),
                "solve_median_ms": statistics.median(s["ms"] for s in solves),
                "solves": solves, "jvp_rhs_shapes": products, "profile": profile}
            rows[variant] = row
            if variant != "pinocchio":
                np.testing.assert_allclose([s["cost"] for s in solves],
                    rows["pinocchio"]["solves"][0]["cost"], rtol=1e-8, atol=1e-10)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(args.label, name, variant, "linearize", round(row["linearize_median_ms"], 2),
                  "solve", round(row["solve_median_ms"], 2), "rhs", products, flush=True)


if __name__ == "__main__":
    main()
