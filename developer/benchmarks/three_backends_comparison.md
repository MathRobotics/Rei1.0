# RoboKots NumPy / Pinocchio / RoboKots Rust timing comparison

Follow-up: this initial comparison forced `jacobian_strategy="mul"` even for
the dense LM solver. It does not represent RoboKots' fastest dense assembly.
See [the performance investigation](rust_performance_investigation.md) for
the isolated cause, fresh native build, and corrected dense measurements.

Measured on 2026-11-05 in Rei's `.venv`: macOS 15.7.4 ARM64, Python 3.13.1,
NumPy 2.5.3, SciPy 1.18.1, Pinocchio 4.1.0, MathRobo 0.0.6,
RoboKots 0.0.1 at `24acfc36a54e889332db798004b37122ce77843d`, and
robokots-native 0.0.1 (native API 1). Rust execution produced
`RustBatchOutwardState` for both models.

These are timings through Rei's trajectory optimization interface, including
its adapters, state updates, trajectory maps, and solver overhead. They do
not measure only the underlying rigid-body dynamics kernels.

## Conditions

- Same URDF per model: `planar2.urdf` (2 DOF), `fr3v2.1_franka_hand.urdf` (7 DOF).
- Same problem: `examples/spec/pinocchio_traj_dynamics.toml`; 201 time samples,
  degree-5 B-splines, 50 control points per joint, gravity `(0, 0, -9.81)`.
- Velocity, acceleration, and torque regularization at **all 201 samples**.
  Torque weight is `1e-10`; this is primarily a motion-smoothing problem.
- RoboKots explicitly selects `numpy` or `rust`, analytic derivatives,
  `jacobian_strategy="mul"`, and `batch_trajectory=True`. Required model order is 3.
- Same nullspace equality elimination and Levenberg–Marquardt solver,
  `max_iters=500`, `tol_dx=1e-12`, starting from the same problem initialization.
- Setup and linearization: 2 warmups, then 7 measured runs. Backend order rotates
  each round. Each linearization uses a new seeded trajectory vector, identical
  across backends, to avoid reusing a cached result at an unchanged point.
- Solve: median of 3 fresh solves per backend, rotating backend order.
  Solve excludes model loading, compilation, and nullspace construction.
  Setup includes model loading and problem compilation, excluding module imports.
- Environment: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`.
  Actual library thread counts were not independently inspected.

## Results

Median wall time; all table values are **milliseconds**.
Linearization computes the full residual and dense Jacobian once.

| Model | Backend | Setup | Linearization | Solve | Solve min–max |
|---|---|---:|---:|---:|---:|
| planar2 (2 DOF) | RoboKots NumPy | 3.44 | 136.71 | 1443.40 | 1399.87–1464.02 |
| planar2 (2 DOF) | Pinocchio | 2.62 | 15.28 | 221.39 | 186.00–330.99 |
| planar2 (2 DOF) | RoboKots Rust | 3.56 | 17.50 | 341.63 | 216.79–360.85 |
| Franka (7 DOF) | RoboKots NumPy | 4.73 | 2509.46 | 24309.71 | 23308.11–25035.86 |
| Franka (7 DOF) | Pinocchio | 3.71 | 30.38 | 667.98 | 664.39–682.57 |
| Franka (7 DOF) | RoboKots Rust | 5.16 | 90.15 | 1238.77 | 1221.98–1295.14 |

Rust reduced solve time relative to RoboKots NumPy by about **4.2x** for
planar2 and **19.6x** for Franka. Pinocchio had the lowest median time in
both cases. For Franka, Pinocchio's solve was about **1.85x** faster than
RoboKots Rust. Planar2 solve ranges overlap, so its small absolute timings
should not be treated as a stable backend ratio.

The first linearization on a fresh runtime was 150.79 / 15.26 / 18.43 ms for
planar2 and 2215.38 / 28.89 / 88.12 ms for Franka (NumPy / Pinocchio / Rust).
These first-runtime measurements occurred after the correctness checks and
exclude process startup/imports; they are not cold-process startup timings.

## Correctness

All 18 solves converged in 9 iterations. Final costs were approximately
`0.415423654342265` (planar2) and `1.454007234781015` (Franka), agreeing across
backends to floating-point roundoff. Dense Jacobian shapes were `(2018, 100)`
and `(7063, 350)` respectively.

Separately, torque regularization was assigned **unit weight**, and residuals
and Jacobians were compared at three random trajectory vectors per model.
All comparisons passed `rtol=atol=1e-8`, preventing the small optimization
torque weight from hiding dynamics discrepancies.

| Model | Compared to Pinocchio | Max absolute torque residual difference | Max absolute Jacobian difference |
|---|---|---:|---:|
| planar2 | RoboKots NumPy | 1.82e-12 | 1.64e-11 |
| planar2 | RoboKots Rust | 9.09e-13 | 1.09e-11 |
| Franka | RoboKots NumPy | 2.27e-12 | 1.53e-10 |
| Franka | RoboKots Rust | 2.27e-12 | 4.37e-11 |

These results apply to this local workload. They do not establish relative
performance for higher torque derivatives, IOC, JAX autodiff, or standalone
native kernels. The older `pinocchio_rust_comparison.md` used a different
problem sampling pattern, solver, and dependency revision; its numbers are
not directly comparable to this run.

## Reproduce

Run from the repository root with the three backends already installed:

```sh
PYTHONPATH=. MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  developer/benchmarks/compare_three_robotics_backends.py \
  --repeat 7 --warmup 2 --solve-repeat 3 --jacobian-strategy mul \
  --output /tmp/rei-three-backends.json
```

The script checks numerical agreement before timing. Raw samples, dependency
versions, the resolved RoboKots source, and solver results are saved in
[three_backends_comparison_results.json](three_backends_comparison_results.json).
