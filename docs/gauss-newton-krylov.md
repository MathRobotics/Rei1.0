# Krylov Gauss–Newton

`gauss_newton_krylov` follows the same outer algorithm as `gauss_newton`:
adaptive damping, Armijo backtracking, retry handling, roundoff-aware
acceptance, and the `tol_grad` convergence check. It replaces only the dense
linear least-squares solve with a preconditioned CGLS iteration using JVP and
VJP products. The default retains a fixed inner tolerance of `1e-10`.
Adaptive inner accuracy is available explicitly with `inner_tol=None`.

```python
from rei import solve

out = solve(runtime, solver="gauss_newton_krylov", options={
    "max_iters": 200,
    "tol_grad": 1e-8,
    "preconditioner": "auto",
    "history_path": "krylov.jsonl",
})
```

When available, the full affine-residual Gram matrix preconditions CGLS,
including correlations between variables. Its eigendecomposition is reused,
with the current damping included when applying the inverse. Otherwise a
fixed number of Rademacher VJP probes estimate a diagonal preconditioner,
refreshed every `preconditioner_refresh` linearizations (default 5).
Preconditioning preserves the damped least-squares problem. Its curvature
scale also supplies an approximate numerical damping floor, avoiding an exact
column scan; affine-only curvature is not a bound on the full nonlinear
Jacobian's scale.

With `inner_tol=None`, the relative normal-residual tolerance is
`max(forcing_min, forcing_max * sqrt(min(1, ||g|| / ||g_initial||)))`, where
`g = J.T @ r`. CGLS checks the true normal residual in the original coordinates
before reporting convergence. An explicit numeric `inner_tol` selects fixed
accuracy. This inner tolerance does not replace the outer `tol_grad` test.
The inner residual is evaluated as `-g - J.T @ (J @ h) - damping*h`.
Keeping the initial gradient separate avoids cancellation from repeatedly
adding a small correction to a large nonzero residual near stationarity.

At fixed accuracy the default inner budget remains `max(20, 2*n)`. In the
optional adaptive mode the budget starts at 50 and doubles after an inner
iteration limit is reached, up to `max(50, 2*n)`. An explicit
`inner_max_iters` is a fixed cap. At the cap, the approximate step is passed
to the common outer line search; a low cap can increase the total number of
outer iterations and the total runtime. On nonconvex problems, looser inner
accuracy can also change which solution is reached.

For comparison or compatibility with earlier behavior, set
`globalization="trust_region"` to use the previous Steihaug PCG trust-region
algorithm. The `initial_radius`, `max_radius`, and `acceptance` options apply
only in that mode. The trust-region mode retains a default fixed inner cap of
50; `forcing_min` and `forcing_max` control adaptive accuracy in both modes.

## Options

All `gauss_newton` options are supported, including damping, tolerances,
line-search controls, history output and callbacks. Krylov-specific options:

| Option | Default | Meaning |
|---|---:|---|
| `inner_tol` | 1e-10 | Fixed CGLS accuracy; `None` explicitly selects adaptive accuracy |
| `inner_max_iters` | `None` | `max(20, 2*n)` at fixed accuracy; starts at 50 and grows in adaptive mode; a number is a fixed cap |
| `forcing_min`, `forcing_max` | 1e-4, 0.1 | Bounds for adaptive inner accuracy |
| `preconditioner` | `auto` | `auto`, `linear`, `diagonal`, or `identity` |
| `preconditioner_probes` | 8 | VJP probes for the diagonal estimate |
| `preconditioner_max_size` | 512 | Variable dimension limit for affine Gram matrices |
| `preconditioner_floor` | 1e-10 | Relative floor for small eigenvalues/diagonal entries |
| `preconditioner_refresh` | 5 | Linearizations between VJP estimate refreshes |
| `seed` | 0 | Seed for reproducible diagonal probes |
| `globalization` | `line_search` | `line_search` or legacy `trust_region` |

The affine Gram path may construct dense affine-only derivative blocks, but the
complete nonlinear Jacobian and its normal matrix are never assembled.
Expression-level fallback derivatives can still be dense if an expression
does not implement products. Probe-based scaling is an estimate, so its
effectiveness and runtime depend on the problem.

The `linear_solve` history events and `out.meta['inner_solves']` record the
actual tolerance, budget, iterations and preconditioner for every direction.
Large inner iteration counts or repeated limit exits indicate that the
current preconditioner is insufficient. Matching the dense solver's outer
line search does not require solving every early linear system to `1e-10`.

See the [measured RoboKots DOC comparison](../developer/benchmarks/kots_doc_krylov.md).
That comparison uses the legacy trust-region mode. The current line-search
implementation has a separate [FR3 regression report](../developer/benchmarks/krylov_fr3_regression.md),
including its remaining total-runtime and convergence limitations.

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
