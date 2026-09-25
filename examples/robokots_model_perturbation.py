"""Compare nominal and perturbed trajectory dynamics at the same trajectory."""
from pathlib import Path
import argparse
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robokots.kots import Kots
from rei import load_problem_spec_toml
from rei.optimize_backends.kots import compile_kots_trajectory_problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perturbation", type=Path,
                        default=ROOT / "examples/perturbation/planar2.toml")
    parser.add_argument("--backend", choices=("numpy", "rust"), default="numpy")
    args = parser.parse_args()
    model = Kots.from_urdf_file(str(ROOT / "examples/models/planar2.urdf"),
                                order=3, backend=args.backend)
    problem = load_problem_spec_toml(ROOT / "examples/spec/robokots_traj_dynamics_d12.toml")
    problem["time"].update(N=10, dt=.2)
    problem["trajectory"]["num_ctrl_points"] = 6
    options = dict(model=model, kots_backend=args.backend, gravity=(0., -9.81, 0.))
    nominal = compile_kots_trajectory_problem(problem, **options)
    perturbed = compile_kots_trajectory_problem(problem, perturbation=args.perturbation, **options)
    point = np.linspace(-.2, .3, nominal.runtime.pack.n_total)
    for compiled in (nominal, perturbed):
        compiled.runtime.pack.set(point)
    r0 = nominal.runtime.eval_stacked_terms(weighted=False)
    r1 = perturbed.runtime.eval_stacked_terms(weighted=False)
    repeated = perturbed.runtime.eval_stacked_terms(weighted=False)
    report = perturbed.perturbation_report
    print(f"seed: {report.seed}")
    print(f"mass scales: {dict(report.mass_scales)}")
    print(f"max raw residual change: {np.max(np.abs(r1-r0)):.6g}")
    print(f"repeat evaluation change: {np.max(np.abs(repeated-r1)):.6g}")


if __name__ == "__main__":
    main()
