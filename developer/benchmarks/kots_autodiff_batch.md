# RoboKots batched JIT AD

Measured on the local CPU with JAX 0.11.2 and RoboKots
`125c8703407fc2863d78df7ad39f5877eefa9de3`:

```sh
PYTHONPATH=. .venv/bin/python developer/benchmarks/kots_autodiff_batch.py
```

Planar 2-DOF model, Rust dynamics, 21 evaluation points, 8 spline control
points, model order 5, torque output (motion order 3). Times cover the complete
DOC residual/Jacobian linearization. Each evaluation changes the parameters
and invalidates the runtime state cache. Warm measurements are medians of
10 evaluations; the first call includes JIT compilation and execution, not an
isolated compiler measurement. All configurations run in one process with
separate models/JIT caches, so process-wide JAX startup costs are not isolated.

| Method | First evaluation | Warm median |
| --- | ---: | ---: |
| Analytic, batch | 9.34 ms | 3.68 ms |
| AD, single-time | 816.86 ms | 19.75 ms |
| AD, batch | 555.88 ms | 4.09 ms |

On this small workload batched AD is about 4.8x faster than single-time AD
after compilation, and about 1.1x the analytic batch time. These are local
measurements, not a guarantee for larger robots or different time grids.

Rei does not cache AD state/Jacobian results. RoboKots caches JIT functions.
DOC-to-IOC completion, KKT stationarity output, dense and operator products,
near-zero motion, sparse/repeated time selection, gravity/order/model changes,
and output-list changes are covered by `tests/test_kots_autodiff_batch.py`.
Those tests forbid analytic derivative API calls on the AD model.
