# Pinocchio / RoboKots Rust comparison

Measured locally on 2026-09-22 using `/Users/a896/.venvs/rei/bin/python`.
Pinocchio 4.1.0, RoboKots 0.0.1, NumPy 2.5.3. The loaded RoboKots model
created a `RustCompiledRobot`; Rust execution was verified.

Both backends use the same URDF, gravity `(0, 0, -9.81)`, initial parameters,
and `examples/spec/pinocchio_traj_dynamics.toml`. This is a Rei trajectory
optimization benchmark, not a standalone RNEA benchmark. The trajectory has
201 samples, degree-5 B-splines, and 50 control points per joint. Torque is
sampled at 5 time points with weight 1e-10. RoboKots order is 5. Equality
constraints are eliminated using the same nullspace reduction.

Times in milliseconds; load/compile/linearize use 7 measured runs after 2
warmups, solve uses the median of 3 fresh solves. Compilation includes model
loading. Solve excludes compilation and nullspace construction. Linearization
uses a different seeded input each time to avoid cache hits. Environment:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` (actual platform BLAS threading was
not independently verified). These short local measurements can fluctuate.

| Model | Backend | Load | Compile | Linearize | Solve |
|---|---|---:|---:|---:|---:|
| planar2 (2 DOF) | Pinocchio | 0.130 | 702.745 | 4.704 | 35.170 |
| planar2 (2 DOF) | RoboKots Rust | 0.465 | 939.346 | 6.311 | 35.681 |
| Franka (7 DOF) | Pinocchio | 0.599 | 670.306 | 36.001 | 339.430 |
| Franka (7 DOF) | RoboKots Rust | 1.993 | 717.026 | 42.719 | 331.066 |

All solves reported `converged` in 2 iterations. Solve timings are effectively
similar here; the few-percent difference does not establish a winner.
Pinocchio linearization was about 1.34x / 1.19x faster in this run.
These findings do not establish performance on torque-dominated objectives,
higher torque derivatives, or inverse optimal control.

Correctness was checked independently with torque weight set to **1**, at
three seeded random trajectory parameter vectors per model. Cross-backend
residual/Jacobian agreement used rtol=atol=1e-8. Both backends' Jacobians
also passed central directional finite differences (epsilon=1e-6,
rtol=atol=1e-5).

| Model | Max absolute torque difference | Max absolute Jacobian difference | Max directional finite-difference error |
|---|---:|---:|---:|
| planar2 | 9.095e-13 | 1.455e-11 | 1.045e-6 |
| Franka | 6.821e-13 | 3.638e-11 | 7.495e-7 |

Compatibility findings:

- The installed RoboKots no longer exposes `state_dict_`. The benchmark now
  passes `None` instead of reading that legacy attribute; Rei's Kots adapter reads
  state through the model API. The subsequent compatibility update also migrates examples/tests and
  makes `data` optional in the Kots builders/compiler.
- `7_dof_arm.urdf` lacks revolute joint limits and is rejected by Pinocchio.
  The unmodified `fr3v2.1_franka_hand.urdf` was used for both 7-DOF runs.
- The benchmark now explicitly aligns gravity and uses absolute parameter
  assignment to avoid accumulated floating-point update errors.

Reproduce from the repository root:

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/a896/.venvs/rei/bin/python developer/benchmarks/verify_robotics_backends.py \
  --output /private/tmp/rei_backend_comparison.json
```

Raw results are stored beside this report in `pinocchio_rust_comparison.json`.
