# RoboKots DOC benchmarks

The production solvers are `gauss_newton`, `gauss_newton_operator` (damped CGLS)
and `gauss_newton_krylov` (preconditioned trust-region CG). The shared runner is
[`kots_doc_operator.py`](kots_doc_operator.py); its filename is retained for
existing reproduction commands, and `--solvers` selects any of these methods.

| Record | Scope | Interpretation |
|---|---|---|
| [CGLS operator comparison](kots_doc_operator.md) | Native Jv/Jᵀv routing and dense parity | CGLS was slower on the measured DOCs |
| [Krylov comparison](kots_doc_krylov.md) | Standard torque DOC and stronger torque weight | Standard cases converged faster; strong-torque cases hit the iteration limit |
| [Torque time derivatives](kots_doc_torque_derivatives.md) | First/second derivatives, separately and together | Includes unconverged cases; 7-DoF figures are single-run reference timings |
| [Further improvement investigation](kots_doc_krylov_investigation.md) | Single-stack fused VJP and preconditioner experiments | Research prototypes only, not enabled in production |

Each report links its `*_results.json` with measured times, solutions, convergence
checks, native API counters and environment information. Compare timings only
under the same conditions. A shorter run that stops before convergence is not
evidence of faster convergence.

```bash
PYTHONPATH=/path/to/RoboKots:. OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python developer/benchmarks/kots_doc_operator.py \
  --model planar2.urdf --steps 201 --controls 50 \
  --solvers gauss_newton gauss_newton_krylov \
  --repeat 3 --warmup 1 --output /tmp/doc-solvers.json
```

Use `--torque-fields torque_d1`, `torque_d2`, or
`torque torque_d1 torque_d2` for derivative residuals. The runner chooses model
order 4 or 5 as needed. `--torque-weight` applies the same numeric weight to each
selected field; these fields have different units.

The investigation runner is [`kots_doc_krylov_investigation.py`](kots_doc_krylov_investigation.py).
It supports `compare`, `verify`, `sweep` and `profile` modes and restores its
process-local experimental patches on exit. It does not edit production files.
Its source-based patch intentionally fails if the relevant runtime code changes;
review the experiment before updating that patch.
