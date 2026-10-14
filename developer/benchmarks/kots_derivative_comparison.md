# Numerical / AD / analytic comparison

Run both stages under the same settings:

```sh
PYTHONPATH=. .venv/bin/python developer/benchmarks/kots_derivative_comparison.py \
  --steps 21 --control-points 8 --output /tmp/rei-kots-comparison.json
```

The report stores the source TOML/model hashes, effective DSL (including
overrides and weights), fixed trajectory, TOML initial values, seed, float64
precision, backend, derivative settings, solver/options, evaluation counts,
final solutions, KKT checks, and full IOC results. Existing output files are
not overwritten. Non-finite or unsupported derivatives fail explicitly.

Stage 1 compares the **unweighted torque Jacobian w.r.t. trajectory parameters**
at exactly the same point. It measures first evaluation and five warm calls
separately. All joint torques are requested in joint-DoF order. Evaluated states
are invalidated before each measurement, including the analytic batch state.
The first AD call includes compilation plus execution, not an isolated compiler
measurement. Configurations use separate model JIT caches but share a process,
so process-wide JAX startup overhead is not isolated.

Stage 2 restores the identical TOML initial point, eliminates the same linear
equalities, and runs DOC→KKT→IOC. DOC timing follows stage-1 JIT warmup; additional
shape-specific compilation, if any, is included. Counters distinguish residual
evaluations, linearizations, dynamics updates and public derivative calls.
The full KKT check uses the original objective/constraint partition; IOC
reports its own stationarity residual and constraint-related diagnostics.

## Local result

CPU, RoboKots `acb2a041453d7c4d3170050e98a1ead81c1897ba`, JAX 0.11.2,
Rust dynamics, planar 2-DOF, 21 samples, 8 spline controls, model order 5,
torque output order 3, gravity `(0, -9.81, 0)`, seed 18. Numerical `eps=1e-5`,
AD `mode=forward, jit=True`, all configurations use `jacobian_strategy=dense`.

| Method | Time batch | First Jacobian | Warm median | Max absolute error vs analytic |
| --- | --- | ---: | ---: | ---: |
| Analytic | off | 10.0 ms | 4.3 ms | 0 |
| Analytic | on | 2.9 ms | 2.5 ms | 0 |
| Numerical | off | 7.3 ms | 6.9 ms | 1.9e-9 |
| Numerical | on | 4.1 ms | 3.1 ms | 1.9e-9 |
| AD | off | 402.4 ms | 10.2 ms | 4.3e-14 |
| AD | on | 541.9 ms | 3.5 ms | 4.3e-14 |

DOC: Gauss–Newton, `tol_grad=1e-8`, `tol_r=1e-10`, `tol_dx=1e-12`,
`max_iters=100`. All six runs converged in 2 iterations, with 3 residual
evaluations and 3 linearizations, objective approximately `0.04406286448`,
and KKT stationarity infinity norm approximately `4.08e-13`.

| Method | Time batch | DOC time | Derivative calls during DOC |
| --- | --- | ---: | ---: |
| Analytic | off | 20.1 ms | 63 |
| Analytic | on | 12.2 ms | 3 |
| Numerical | off | 27.9 ms | 63 |
| Numerical | on | 14.1 ms | 3 |
| AD | off | 37.6 ms | 63 |
| AD | on | 15.2 ms | 3 |

All IOC results agree to numerical precision, with weights approximately
`[0.98445524, 0.01240508, 0.00313968]`. IOC stationarity norm is about **41.04**:
the default IOC model here excludes DOC constraints, as its diagnostics report.
This comparison verifies agreement of evaluation paths, not recovery of the
original weights or satisfaction of constrained IOC stationarity. Timings
describe this small workload and are not general speedup guarantees.

The complete recorded run is in `kots_derivative_comparison_results.json`.
Torque time derivatives `torque_d1`–`torque_d3` are additionally tested against
single-time and analytic evaluation; those higher-order cases are not part of
the timing table. Fixed joints, nonzero gravity, near-zero states, selected
times/outputs, JVP/VJP and public `jit/mode/eps` options are regression-tested.
