"""DOC -> noisy joint observations -> B-spline fit -> IOC comparison."""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robokots.kots import Kots
from rei import load_problem_spec_toml, prepare_noisy_ioc_trajectory, solve
from rei.optimize.reductions import build_nullspace_equality_reduction
from rei.optimize_backends.kots import compile_kots_trajectory_problem
from rei.optimize_backends.trajectory_ioc import estimate_ioc_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-std", type=float, default=.001, help="Joint angle standard deviation in radians.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("numpy", "rust"), default="rust")
    parser.add_argument("--output", type=Path, help="Save observations, fitted parameters and IOC weights as NPZ.")
    args = parser.parse_args()
    if not np.isfinite(args.noise_std) or args.noise_std < 0 or args.seed < 0:
        parser.error("noise-std must be finite and nonnegative; seed must be nonnegative")
    spec = load_problem_spec_toml(ROOT / "examples/spec/robokots_traj_dynamics_d12.toml")
    spec["time"].update(N=20, dt=.1)
    spec["trajectory"]["num_ctrl_points"] = 8

    def compile_problem():
        model = Kots.from_urdf_file(str(ROOT / "examples/models/planar2.urdf"), order=3, backend=args.backend)
        return compile_kots_trajectory_problem(spec, model=model, kots_backend=args.backend)

    doc = compile_problem()
    reduction = build_nullspace_equality_reduction(doc.runtime, eq_selector_attr="enforce", eq_selector_value="nullspace")
    outcome = solve(reduction.runtime, options={"verbose": False, "tol_grad": 1e-8, "max_iters": 200})
    print(f"DOC status: {outcome.status}")
    if not outcome.converged:
        raise SystemExit("DOC did not converge; IOC was not run.")
    full_point = reduction.lift(outcome.solution)
    start, stop = doc.runtime.pack.slices[doc.p_var]
    clean_p = full_point[start:stop].copy()
    observed = prepare_noisy_ioc_trajectory(doc.trajectory_map, clean_p, std=args.noise_std, seed=args.seed)
    ioc = compile_problem()
    clean_result = estimate_ioc_weights(ioc, p=full_point)
    noisy_point = full_point.copy()
    noisy_point[start:stop] = observed.p
    noisy_result = estimate_ioc_weights(ioc, p=noisy_point)
    print(f"noise std (rad): {args.noise_std}, seed: {args.seed}")
    print(f"fit residual norm: {observed.fit_residual_norm:.6g}, rank: {observed.fit_rank}")
    print(f"clean IOC weights: {clean_result['weights']}")
    print(f"noisy IOC weights: {noisy_result['weights']}")
    print(f"stationarity residual (clean / noisy): "
          f"{clean_result['stationarity']['ikkt_residual_norm']:.6g} / "
          f"{noisy_result['stationarity']['ikkt_residual_norm']:.6g}")
    for message in noisy_result["validation"]["messages"]:
        print(f"IOC diagnostic: {message}")
    if args.output:
        np.savez(args.output, p_clean=clean_p, p_noisy=observed.p, q_clean=observed.q_clean,
                 q_observed=observed.q_observed, q_fitted=observed.q_fitted, noise=observed.noise,
                 std=observed.std, seed=args.seed, dt=doc.dt,
                 weights_clean=clean_result["weights"], weights_noisy=noisy_result["weights"])


if __name__ == "__main__":
    main()
