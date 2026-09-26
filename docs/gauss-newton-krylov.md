# Krylov Gauss–Newton

`gauss_newton_krylov` follows the same outer algorithm as `gauss_newton`:
adaptive damping, Armijo backtracking, retry handling, roundoff-aware
acceptance, and the `tol_grad` convergence check. It replaces only the dense
linear least-squares solve with a diagonally scaled CGLS iteration using JVP
and VJP products.

```python
from rei import solve

out = solve(runtime, solver="gauss_newton_krylov", options={
    "max_iters": 200,
    "tol_grad": 1e-8,
    "preconditioner": "auto",
    "history_path": "krylov.jsonl",
})
```

The scale is estimated from the diagonal of an available affine-residual Gram
matrix or from a fixed number of Rademacher VJP probes. Affine curvature is
constant and reused; probe estimates refresh every `preconditioner_refresh`
linearizations (default 5). It changes the coordinates used by CGLS, not the damped least-squares
problem being solved. The numerical damping floor uses the corresponding
diagonal scale estimate, avoiding an exact column scan. Inner solves report
their convergence status; when `inner_max_iters` is reached, the approximate
step is still passed to the common outer line search.

For comparison or compatibility with earlier behavior, set
`globalization="trust_region"` to use the previous Steihaug PCG trust-region
algorithm. Its `initial_radius`, `max_radius`, `acceptance`, `forcing_min`, and
`forcing_max` options apply only in that mode.

## Options

All `gauss_newton` options are supported, including damping, tolerances,
line-search controls, history output and callbacks. Krylov-specific options:

| Option | Default | Meaning |
|---|---:|---|
| `inner_tol` | 1e-10 | Relative scaled normal-residual tolerance for CGLS |
| `inner_max_iters` | `None` | Maximum CGLS iterations; defaults to `max(20, 2*n)` |
| `preconditioner` | `auto` | `auto`, `linear`, `diagonal`, or `identity` |
| `preconditioner_probes` | 8 | VJP probes for the diagonal estimate |
| `preconditioner_max_size` | 512 | Variable dimension limit for affine Gram matrices |
| `preconditioner_floor` | 1e-10 | Relative floor for small diagonal entries |
| `preconditioner_refresh` | 5 | Linearizations between VJP estimate refreshes |
| `seed` | 0 | Seed for reproducible diagonal probes |
| `globalization` | `line_search` | `line_search` or legacy `trust_region` |

The affine Gram path may construct dense affine-only derivative blocks, but the
complete nonlinear Jacobian and its normal matrix are never assembled.
Expression-level fallback derivatives can still be dense if an expression
does not implement products. Probe-based scaling is an estimate, so its
effectiveness and runtime depend on the problem.

See the [measured RoboKots DOC comparison](../developer/benchmarks/kots_doc_krylov.md).

## Torque time derivatives

RoboKots trajectory residuals can use `torque_d1` or `torque_d2` via
`quantity = { name = "joint_torques", field = "torque_d1" }` in a TOML term.
Use a model with `order=4` for the first derivative and `order=5` for the second;
the trajectory must also supply the corresponding higher motion derivatives.
The DOC benchmark selects the required model order automatically:

```bash
python developer/benchmarks/kots_doc_operator.py \
  --steps 201 --controls 50 --torque-fields torque torque_d1 torque_d2 \
  --solvers gauss_newton gauss_newton_krylov --output /tmp/torque-derivatives.json
```

Every selected field receives the torque term's weight (or `--torque-weight`).
These quantities have different units; equal numeric weights are a benchmark
convention, not a recommendation for physical tuning.

See [torque-derivative DOC timings](../developer/benchmarks/kots_doc_torque_derivatives.md)
for measurements and the convergence limits observed at 201 points / 50 controls.
