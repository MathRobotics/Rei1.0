"""Compare backend timings and verify unweighted torque derivatives."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from developer.benchmarks.urdf_rust_kots_vs_pinocchio import (
    ROOT,
    compile_problem,
    load_problem_spec_toml,
    measure_backend,
    set_runtime_point,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(tempfile.gettempdir()) / "rei_backend_comparison.json",
    )
    args = parser.parse_args()
    result = {
        "versions": {name: version(name) for name in ("pin", "robokots", "numpy")},
        "platform": platform.platform(),
        "models": {},
    }
    problem = load_problem_spec_toml(ROOT / "examples/spec/pinocchio_traj_dynamics.toml")
    for name in ("planar2", "fr3v2.1_franka_hand"):
        path = ROOT / "examples/models" / f"{name}.urdf"
        rows = {}
        for backend in ("pinocchio", "kots-rust"):
            rows[backend] = measure_backend(
                backend=backend, problem=problem, urdf_path=path, order=5,
                repeat=7, warmup=2, solve_repeat=3,
            )
            print(name, backend, rows[backend], flush=True)

        # Isolate torque so its tiny weight cannot hide discrepancies.
        torque = copy.deepcopy(problem)
        torque["terms"] = [copy.deepcopy(problem["terms"][-1])]
        torque["terms"][0]["cost"] = {"type": "scalar_weight", "w": 1.0}
        compiled = [
            compile_problem(backend, torque, path, order=5)
            for backend in ("pinocchio", "kots-rust")
        ]
        rng = np.random.default_rng(42)
        errors = {
            "torque_max_abs": 0.0,
            "jacobian_max_abs": 0.0,
            "directional_fd_max_abs": 0.0,
        }
        for _ in range(3):
            x = rng.normal(0, 0.05, compiled[0].runtime.pack.n_total)
            values = []
            for pc in compiled:
                set_runtime_point(pc.runtime, x)
                residual, jacobian = pc.runtime.linearize()
                values.append((np.asarray(residual).copy(), np.asarray(jacobian).copy()))
            for index, key in enumerate(("torque_max_abs", "jacobian_max_abs")):
                errors[key] = max(
                    errors[key], float(np.max(np.abs(values[0][index] - values[1][index])))
                )
                np.testing.assert_allclose(
                    values[0][index], values[1][index], rtol=1e-8, atol=1e-8,
                )
            direction = rng.normal(size=x.size)
            direction /= np.linalg.norm(direction)
            for pc, (_, jacobian) in zip(compiled, values, strict=True):
                set_runtime_point(pc.runtime, x + 1e-6 * direction)
                plus = pc.runtime.linearize()[0].copy()
                set_runtime_point(pc.runtime, x - 1e-6 * direction)
                minus = pc.runtime.linearize()[0].copy()
                finite_difference = (plus - minus) / 2e-6
                errors["directional_fd_max_abs"] = max(
                    errors["directional_fd_max_abs"],
                    float(np.max(np.abs(finite_difference - jacobian @ direction))),
                )
                np.testing.assert_allclose(
                    finite_difference, jacobian @ direction, rtol=1e-5, atol=1e-5,
                )
        model = compiled[1].state_builder.model
        errors["rust_robot_type"] = str(type(model._rust_compiled_robot_))
        rows["correctness"] = errors
        result["models"][name] = rows
        print(name, errors, flush=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
