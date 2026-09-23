"""Audit first/second-order conditions separately from solver status."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rei import compile_nls_problem_spec, load_problem_spec_toml, solve
from rei.optimize.kkt import check_kkt_conditions, check_kkt_residuals
from rei.optimize.reductions import build_nullspace_equality_reduction
from rei.xops import set_pack_x


class ScalarResidual:
    n_total = 1

    def __init__(self, residual, jacobian, initial=0.0):
        self.x = np.array([initial])
        self.residual = residual
        self.jacobian = jacobian

    def get_point(self):
        return self.x.copy()

    def set_point(self, x):
        self.x = np.asarray(x, dtype=float).copy()

    def required_list(self, required=None):
        return []

    def eval(self, *, required=None):
        return np.array([self.residual(self.x[0])])

    def linearize(self, *, required=None):
        return self.eval(), np.array([[self.jacobian(self.x[0])]])


def analytic_cases():
    results = {}
    for name, problem in {
        "convex_quadratic": ScalarResidual(lambda x: x - 2, lambda x: 1),
        "nonminimum_stationary_point": ScalarResidual(lambda x: x*x - 1, lambda x: 2*x),
        "small_residual_large_gradient": ScalarResidual(lambda x: 1e12*x + 1e-11, lambda x: 1e12),
    }.items():
        out = solve(problem)
        r, j = problem.linearize()
        results[name] = {"status": out.status, "x": out.solution.tolist(),
                         "cost": float(r @ r), "half_cost_gradient": float((j.T @ r)[0])}
    # These two cases share the objective (x-2)^2 and constraint x<=1 or x=1.
    for kind in ("eq", "ineq"):
        constraint = {"name": "limit", "var": "x", "kind": kind, "weight": 100.0}
        if kind == "eq":
            constraint["target"] = [1.0]
        else:
            constraint["bounds"] = {"upper": [1.0]}
        runtime = compile_nls_problem_spec(
            {"opt_vals": {"x": {"init": [0.0]}}, "terms": [
                {"name": "objective", "var": "x", "target": [2.0]}, constraint,
            ]}, build_state=lambda *args, **kwargs: {},
        )
        out = solve(runtime)
        kkt = check_kkt_conditions(runtime)
        results[f"{kind}_penalty"] = {
            "status": out.status, "x": out.solution.tolist(), "kkt_ok": kkt.ok,
            "eq_violation": kkt.eq_violation_inf, "ineq_violation": kkt.ineq_violation_inf,
        }
        if kind == "eq":
            reduction = build_nullspace_equality_reduction(runtime)
            reduced = solve(reduction.runtime)
            reduction.runtime.update_state_if_needed()
            results["equality_nullspace"] = {
                "status": reduced.status, "x": runtime.pack.get().tolist(),
                "kkt_ok": check_kkt_conditions(runtime).ok,
            }
        else:
            set_pack_x(runtime.pack, [1.0])
            reported = check_kkt_conditions(runtime)
            raw = check_kkt_residuals(
                grad_objective=[-1.0], ineq_residual=[0.0], ineq_jacobian=[[1.0]],
            )
            results["inequality_exact_boundary"] = {
                "runtime_kkt_ok": reported.ok, "raw_margin_kkt_ok": raw.ok,
                "runtime_stationarity": reported.stationarity_inf,
            }
    return results


def trajectory_case(backend):
    from developer.benchmarks.urdf_rust_kots_vs_pinocchio import ROOT, compile_problem

    problem = load_problem_spec_toml(ROOT / "examples/spec/pinocchio_traj_dynamics.toml")
    compiled = compile_problem(backend, problem, ROOT / "examples/models/planar2.urdf", order=5)
    runtime = compiled.runtime
    reduction = build_nullspace_equality_reduction(
        runtime, eq_selector_attr="enforce", eq_selector_value="nullspace",
    )
    reduced = reduction.runtime
    out = solve(reduced, options={"max_iters": 500, "tol_dx": 1e-8})
    reduced.update_state_if_needed()
    z = reduced.pack.get().copy()
    r, j = reduced.linearize()
    cost = float(r @ r)
    gradient = j.T @ r
    spectra = []
    # Hessian of half the weighted squared residual norm, in reduced coordinates.
    try:
        for epsilon in (1e-5, 2e-5):
            hessian = np.empty((z.size, z.size))
            for col in range(z.size):
                delta = np.zeros(z.size)
                delta[col] = epsilon
                set_pack_x(reduced.pack, z + delta)
                rp, jp = reduced.linearize()
                gp = jp.T @ rp
                set_pack_x(reduced.pack, z - delta)
                rm, jm = reduced.linearize()
                hessian[:, col] = (gp - jm.T @ rm) / (2 * epsilon)
            spectra.append({"epsilon": epsilon,
                            "min_eigenvalue": float(np.linalg.eigvalsh((hessian+hessian.T)/2)[0]),
                            "asymmetry_max": float(np.max(np.abs(hessian-hessian.T)))})
    finally:
        set_pack_x(reduced.pack, z)
        reduced.update_state_if_needed()
    kkt = check_kkt_conditions(runtime)
    max_angle = max(
        float(np.max(np.abs(compiled.trajectory_map.q_at(runtime.pack.get(), k))))
        for k in range(compiled.trajectory_map.steps)
    )
    restarts = []
    rng = np.random.default_rng(23)
    try:
        for scale in (0.1, 1.0, 3.0):
            set_pack_x(reduced.pack, z + scale * rng.normal(size=z.size))
            trial = solve(reduced, options={"max_iters": 500, "tol_dx": 1e-8})
            reduced.update_state_if_needed()
            trial_kkt = check_kkt_conditions(runtime)
            restarts.append({"perturbation_scale": scale, "status": trial.status,
                             "cost": trial.stats.objective, "kkt_ok": trial_kkt.ok,
                             "stationarity_inf": trial_kkt.stationarity_inf})
    finally:
        set_pack_x(reduced.pack, z)
        reduced.update_state_if_needed()
    return {"status": out.status, "iterations": out.stats.iterations, "cost": cost,
            "reduced_dim": z.size, "reduced_gradient_inf": float(np.max(np.abs(gradient))),
            "kkt_ok": kkt.ok, "kkt_stationarity_inf": kkt.stationarity_inf,
            "eq_violation_inf": kkt.eq_violation_inf,
            "ineq_violation_inf": kkt.ineq_violation_inf, "reduced_hessian": spectra,
            "sampled_joint_bound_margin": 3.14159 - max_angle, "restarts": restarts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/rei-forward-optimality.json"))
    args = parser.parse_args()
    results = {"analytic": analytic_cases(), "trajectory": {}}
    print(json.dumps(results["analytic"], indent=2), flush=True)
    for backend in ("pinocchio", "kots-rust"):
        results["trajectory"][backend] = trajectory_case(backend)
        print(backend, results["trajectory"][backend], flush=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
