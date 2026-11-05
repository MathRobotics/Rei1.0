# Rust performance investigation, 2026-11-05

The initial three-backend comparison forced `jacobian_strategy="mul"` while
using a dense Levenberg–Marquardt solver. That configuration makes Rei send
all spline parameter columns through the directional derivative API. Using
the dedicated dense motion Jacobian substantially reduces the Rust cost in
this workload. The installed wheel and a fresh release build show the same
behavior; rebuilding alone does not remove the bottleneck.

## Controlled comparison on the same problem

Same settings as [the initial comparison](three_backends_comparison.md):
201 evaluation points, 50 controls/joint, analytic derivatives, ordinary
joint torque, fixed-base Franka URDF, gravity `(0, 0, -9.81)`, nullspace
reduction and LM. Only the Jacobian strategy changes. Median of 7
linearizations after 2 warmups and 3 fresh solves. All solves converge in
9 iterations to cost `1.454007234781015` within roundoff.

| Native build | Backend / strategy | Linearization (ms) | Solve (ms) |
|---|---|---:|---:|
| Installed wheel | Pinocchio | 36.44 | 762.36 |
| Installed wheel | RoboKots Rust, `mul` | 94.41 | 1478.73 |
| Installed wheel | RoboKots Rust, `dense` | 38.57 | 768.48 |
| Fresh release | Pinocchio | 38.45 | 910.97 |
| Fresh release | RoboKots Rust, `mul` | 96.40 | 1461.53 |
| Fresh release | RoboKots Rust, `dense` | 40.21 | 893.91 |

The installed-wheel run improves about 2.45x in linearization and 1.92x in
solve time by choosing `dense`. Rust and Pinocchio solve times are effectively
similar with this choice. Absolute timings vary between runs, so differences
of a few percent should not be interpreted as a backend or build advantage.

### Why `mul` costs more here

For 7 DOF and model order 3 there are **21 motion-coordinate inputs**, but
**350 spline parameters**. Instrumentation observes one JVP call with RHS
shape **`(201, 21, 350)`**. The `mul` strategy propagates all 350 directions
through the native selected-output workspace to construct a dense matrix.

With `dense`, RoboKots forms a small motion Jacobian per sample and Rei
chains it with the spline maps. Profiling the installed wheel records:

- `mul`: 117 ms profiled total, 68 ms in Rei's batched parameter-Jacobian
  assembly, including 54 ms inside native `RustSelectedWorkspace.apply`.
- `dense`: 59 ms profiled total, 10 ms in the batched parameter-Jacobian
  assembly. About 20 ms remains in motion/map composition and another 20 ms
  in residual/Jacobian expression evaluation.

Profiler timings include instrumentation overhead and are separate from
the unprofiled timing table. These measurements also show why an improvement
in a standalone Rust kernel need not produce the same improvement in the
full Rei solve.

The same effect appears in NumPy. A fresh three-backend `dense` run gives
LM solve medians of 0.713 / 0.214 / 0.200 seconds for planar2 and
8.554 / 1.061 / 0.870 seconds for Franka (NumPy / Pinocchio / Rust).
Franka linearization samples in that run fluctuate substantially: Rust
29.5–92.0 ms and Pinocchio 31.6–126.3 ms. The controlled table above is the
more useful comparison of the assembly strategies; the raw results retain
both experiments rather than mixing their samples.

## What changed from the older performance report

The [2026-09-22 report](pinocchio_rust_comparison.md) used Gauss–Newton,
2 iterations and 5 torque evaluation points. The initial new comparison
used LM, 9 iterations and 201 torque evaluation points. Current-model
RoboKots order is 3; the older report specified order 5. Rei's DSL,
trajectory adapters and dependencies have also changed, so the old absolute
numbers are not a same-code regression baseline.

To isolate the main workload changes, the fresh Rust build was measured
with current Rei and `dense`, holding the rest of the current problem fixed:

| Solver | Torque evaluation points | Iterations | Solve median (ms) |
|---|---:|---:|---:|
| Gauss–Newton | 5 | 2 | 132.79 |
| Gauss–Newton | 201 | 2 | 211.06 |
| LM | 5 | 9 | 549.24 |
| LM | 201 | 9 | 893.91 |

The solver/iteration difference, torque sampling and the forced wide JVP
account for major differences in the reported end-to-end time. This audit
does not establish a regression or improvement against an identically built
historical native binary.

## Source changes and diagnostic controls

RoboKots now shares selected native workspaces across ordinary and high-order
torque Jacobians/products. Commit `d2f8318` shares/caches selections, and
`32f0b0b` implements dedicated dense dynamics Jacobians. The older specialized
order-3 torque API remains available as a fallback.

As a diagnostic, selected-output dispatch was temporarily bypassed in the
benchmark process for ordinary joint torque, forcing this existing fallback.
For Franka the installed-wheel fallback gives 140.35 ms (`mul`) and 39.86 ms
(`dense`), compared with 94.41 / 38.57 ms via selected dispatch. The fresh
build gives the same trend. Thus forcing the older-style torque path does
not improve this case. This control changes dispatch only; it is not a
reproduction of the entire old Python/Rust stack. No dispatch patch was
applied to either source repository.

The latest local `robokots-rs` working tree (base `daeb48e`, including its
uncommitted changes) was copied to `/tmp/rei-native-audit` and built with
`maturin build --offline --release`. The resulting wheel was extracted and
selected with `PYTHONPATH`; the installed project wheel was not replaced.
All timing runs used for the report were performed after compilation finished.
The two native binaries have different SHA-256 hashes, recorded in the raw
results. Source fingerprint and wheel hash are recorded there as well;
version `0.0.1` alone does not identify a build.

## Changes and validation

- The trajectory example now defaults to `dense` for dense solvers and `mul`
  for operator/Krylov solvers. An explicit `--jacobian-strategy` still overrides it.
- The three-backend benchmark exposes `--jacobian-strategy` and defaults to
  `dense` for its dense LM solver. Its original `mul` report is retained with
  an explicit reproduction flag and a link to this correction.
- Core adapter defaults and direct JVP/VJP APIs are unchanged.
- Every diagnostic variant checks residuals and Jacobians against Pinocchio,
  including torque rows rescaled to unit weight, and checks final objective
  agreement. Specialized fallbacks are used only in the diagnostic process.
- The fresh native build passes **78 regression tests**, with `--fail-on-skip`:
  compile order, URDF/JSON parity, perturbation and joint locking.

## Reproduce

Run serially, without a concurrent build or benchmark:

```sh
PYTHONPATH=. MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  developer/benchmarks/investigate_rust_performance.py \
  --label installed-wheel --output /tmp/rei-rust-installed-audit.json
```

For a separately extracted fresh wheel, prepend its extraction directory to
`PYTHONPATH`. To isolate the older workload settings, add
`--model fr3v2.1_franka_hand --variant pinocchio --variant selected-mul
--variant selected-dense --solver gauss_newton --torque-step 50`.

For the corrected three-backend comparison:

```sh
PYTHONPATH=. MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  developer/benchmarks/compare_three_robotics_backends.py \
  --jacobian-strategy dense --output /tmp/rei-three-backends-dense.json
```

[Raw timings, profiles, checks and build fingerprints](rust_performance_investigation_results.json).
