# Python compatibility verification

Verified on macOS ARM64 on 2026-09-22 against the current working tree.

| Dependency | Python 3.11.14 | Python 3.13.1 |
|---|---|---|
| NumPy | 2.4.6 | 2.5.3 |
| SciPy | 1.17.1 | 1.18.1 |
| Pinocchio (`pin`) | 4.1.0 | 4.1.0 |
| pytest | 9.1.1 | 9.1.1 |
| MathRobo | 0.0.4 | 0.0.4 |
| RoboKots | 0.0.1 | 0.0.1 |
| Full test suite | 438 passed | 438 passed |

Both environments used RoboKots commit
`3f2673d7af58cd34f7af11686136241cdf905c91`. Dependency checks passed.
On Python 3.11, the Rust extension was built and imported, and the
stationarity IOC, RoboKots IK, trajectory dynamics, and both-backend
trajectory/IOC comparison examples completed successfully.

Python 3.11 remains the declared minimum. This verification does not cover
other operating systems or the optional native cyipopt dependency.

To reproduce with the same local lock file, select a separate environment:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/rei-py311/venv uv sync --frozen --python 3.11 \
  --group dev --group kots --group pinocchio --group solver-liteopt \
  --extra solver-scipy --link-mode copy
PYTHONPATH=. MPLBACKEND=Agg /tmp/rei-py311/venv/bin/python -m pytest -q -ra
```

`uv.lock` is currently ignored by Git, and the RoboKots source follows
`develop`. A new resolution can therefore select different dependency
versions; the table and commit above record the environment actually tested.
