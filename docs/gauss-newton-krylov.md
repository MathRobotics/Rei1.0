# Krylov Gauss–Newton

`gauss_newton_krylov` is a separate NumPy solver designed for residual JVP/VJP
products. Existing Gauss–Newton and CGLS operator solvers are unchanged.

```python
from rei import solve

out = solve(runtime, solver="gauss_newton_krylov", options={
    "max_iters": 200,
    "tol_grad": 1e-8,
    "preconditioner": "auto",
    "history_path": "krylov.jsonl",
})
```

Each outer trial approximately minimizes `||r + J h||²` inside a Euclidean
trust region. Preconditioned Steihaug CG applies `J.T @ (J @ p)` through
products, stops on the trust boundary or its residual tolerance, and permits
inexact steps at the iteration limit. The forcing tolerance tightens as the
outer gradient decreases. Actual versus predicted reduction controls acceptance
and the radius. Near floating-point objective resolution, an independently
computed gradient must converge or decrease by at least a factor of two before
a trial is accepted. Rejected trials and evaluation exceptions restore the
accepted point.

There is no damping-floor column scan. Convergence means only
`max(abs(J.T @ r)) <= tol_grad`; a tiny radius or unrepresentable step reports
`stalled`, never success. `max_iters` counts all trials, including rejections.

## Preconditioning

- `auto` (default): use constant affine-residual curvature when available and
  within the size budget; otherwise estimate a diagonal with VJP probes.
- `linear`: require the affine-residual preconditioner.
- `diagonal`: average `(J.T @ w)**2` for independent Rademacher residual vectors.
- `identity`: no preconditioning or probes.

The affine path recognizes built-in variable/trajectory expressions, constants,
subtraction, stacking, components and time differences with constant built-in
weights. It excludes state expressions, hinges and robust costs. DOC velocity
and acceleration penalties therefore supply curvature without differentiating
dynamics. Nullspace reduction projects this curvature into reduced coordinates.
It builds dense **affine-only** derivative blocks and a Gram matrix, then caches
its regularized eigendecomposition. This path is not entirely matrix-free;
`preconditioner_max_size=512` bounds the variable dimension in both the full and
reduced runtime. Larger problems fall back to the diagonal estimate. The
complete nonlinear Jacobian and its normal matrix are never requested by the
solver. Expression-level fallback derivatives can still be dense if an
expression has no product implementation.

The diagonal estimate is only a preconditioner, not a certified norm bound or
convergence test. Its default eight probes do not grow with variable count.
`seed=0` makes them reproducible. Weak affine curvature or poor diagonal
estimates can require many products; speedups are problem dependent.

## Options and output

| Option | Default | Meaning |
|---|---:|---|
| `max_iters` | 200 | All outer trials |
| `tol_grad` | 1e-8 | Infinity norm of `J.T r` |
| `inner_max_iters` | 50 | Maximum CG steps per trial |
| `forcing_min`, `forcing_max` | 1e-4, 0.1 | Relative inner residual tolerance range |
| `initial_radius` | `None` | Automatic scale from preconditioned gradient |
| `max_radius` | 1e8 | Radius upper bound |
| `acceptance` | 0.1 | Minimum gain ratio away from roundoff |
| `preconditioner_probes` | 8 | VJP probes per diagonal estimate |
| `preconditioner_max_size` | 512 | Affine Gram variable budget |
| `preconditioner_floor` | 1e-10 | Relative eigenvalue/diagonal floor |
| `preconditioner_refresh` | 5 | Accepted steps between diagonal refreshes |

The existing `x0`, `required`, `weighted`, `term_indices`, `on_iter`, `profiler`,
`history`, `history_vectors`, `history_path` and `verbose` interfaces are
supported. Generic models need point/state methods and `eval`, `jvp`, `vjp`;
they need no `linearize` method. They may optionally provide a constant,
positive-semidefinite `linear_residual_gram(max_size=...)` in their solver
coordinates. Weighting and selection must already be reflected in that matrix.

Use `trial_history_path` for trust-region trials. Its automatic filename is
`<history stem>.trust_region_trials.jsonl`. `out.trial_history` contains rejected
and accepted trials; `out.meta["inner_solves"]` contains CG diagnostics and
`out.meta["preconditioner"]` records the selected preconditioner. Inner boundary
and iteration-limit exits are explicitly distinguished from inner convergence.
Old damping, line-search, `inner_tol` and `tol_dx` options are rejected.

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
