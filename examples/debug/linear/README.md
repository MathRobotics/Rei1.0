# Linear Trajectory Debug Utilities

This folder contains a debug utility for linear trajectory DSL (`[trajectory].type = "linear"`).

The script loads a TOML file with `[trajectory]`, builds the linear trajectory map, and writes a debug plot PNG.

## Run

```bash
PYTHONPATH=. MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp \
  .venv/bin/python examples/debug/linear/debug_linear_traj.py \
  --dsl examples/debug/linear/linear_traj_demo.toml \
  --output examples/debug/linear/out/linear_debug.png \
  --check-jacobian
```

## Main options

- `--steps`: Override default trajectory steps.
- `--q-dim`: Override default trajectory q-dim.
- `--p`: Override decision vector `p` with comma-separated values.
- `--show`: Display a plot window in addition to saving.
- `--check-jacobian`: Validate `dq/dp` by finite differences.

## Outputs

- Terminal diagnostics:
  - shape/rank/condition estimate of `A`
  - consistency check for `A @ p + b` vs `q_at(p, k)`
  - optional finite-difference Jacobian error
- Plot image:
  - trajectory (`q[0]-q[1]` if `q_dim >= 2`)
  - each `q[j]` over `k`
  - heatmap of matrix `A`
  - per-step block norms (`||A_k||_F`, `||b_k||_2`)
