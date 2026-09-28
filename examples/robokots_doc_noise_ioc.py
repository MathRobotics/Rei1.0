"""DOC -> noisy joint observations -> B-spline fit -> IOC comparison."""
import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robokots.kots import Kots
from robokots.urdf_io import urdf_xml_to_model_data
from rei.backends.state.robotics.urdf import load_joint_locks_toml, lock_urdf_joints
from rei import load_problem_spec_toml, prepare_noisy_ioc_trajectory, solve
from rei.optimize.reductions import build_nullspace_equality_reduction
from rei.optimize_backends.kots import compile_kots_trajectory_problem
from rei.optimize_backends.trajectory_ioc import estimate_ioc_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-std", type=float, default=.001, help="Joint angle standard deviation in radians.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("numpy", "rust"), default="rust")
    parser.add_argument("--model", type=Path, default=ROOT / "examples/models/planar2.urdf")
    parser.add_argument("--spec", type=Path, default=ROOT / "examples/spec/robokots_traj_dynamics_d12.toml")
    parser.add_argument("--gravity", type=float, nargs=3, metavar=("GX", "GY", "GZ"),
                        help="World-frame gravity in m/s^2; otherwise use the backend default.")
    parser.add_argument("--lock-joint", action="append", default=[], metavar="NAME=POSITION",
                        help="Lock a joint in radians/metres, overriding model.locked_joints in --spec. Repeatable.")
    parser.add_argument("--output", type=Path, help="Save observations, fitted parameters and IOC weights as NPZ.")
    args = parser.parse_args()
    if not np.isfinite(args.noise_std) or args.noise_std < 0 or args.seed < 0:
        parser.error("noise-std must be finite and nonnegative; seed must be nonnegative")
    try:
        locks = load_joint_locks_toml(args.spec)
        cli_names = set()
        for item in args.lock_joint:
            name, value = item.split("=", 1)
            if name in cli_names:
                raise ValueError(f"Duplicate locked joint: {name}")
            cli_names.add(name)
            locks[name] = float(value)
        reduced = lock_urdf_joints(args.model.read_text(), locks)
        # This RoboKots version does not enforce movable mimic relationships.
        if ET.fromstring(reduced.xml).find("joint/mimic") is not None:
            raise ValueError("Movable mimic joints are unsupported by this example; lock their masters.")
    except (ValueError, ET.ParseError) as exc:
        parser.error(str(exc))
    model_data = urdf_xml_to_model_data(reduced.xml)
    spec = load_problem_spec_toml(args.spec)
    spec["time"].update(N=20, dt=.1)
    spec["trajectory"]["num_ctrl_points"] = 8

    def compile_problem():
        model = Kots.from_json_data(model_data, order=3, backend=args.backend)
        return compile_kots_trajectory_problem(spec, model=model, kots_backend=args.backend, gravity=args.gravity)

    doc = compile_problem()
    print(f"Model DOF: {doc.trajectory_map.q_dim}; locked joints: {reduced.locked_joints}")
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
