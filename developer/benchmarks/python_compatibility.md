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

To reproduce the current committed lock (not the historical environment above),
select a separate environment:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/rei-py311/venv uv sync --locked --python 3.11 \
  --group dev --group kots --group pinocchio --group solver-liteopt \
  --extra solver-scipy --link-mode copy
PYTHONPATH=. MPLBACKEND=Agg /tmp/rei-py311/venv/bin/python -m pytest -q -ra
```

As of the remote integration on 2026-10-14, `uv.lock` is committed and RoboKots is pinned to
`acb2a041453d7c4d3170050e98a1ead81c1897ba`. Use `--locked` to reject
unintentional dependency resolution. The table above remains a historical record;
it is not a test report for the new revision. See [testing instructions](../../docs/testing.md).

## Verification on 2026-10-14 (before remote integration)

Fresh environments were installed from the committed lock with uv 0.9.18 on
macOS 15.7.4 ARM64. The existing project `.venv` was left unchanged.

| Dependency / check | Python 3.11.14 (core + dev) | Python 3.13.1 (dev + robotics) |
|---|---|---|
| NumPy | 2.4.6 | 2.5.3 |
| pytest | 9.1.1 | 9.1.1 |
| Matplotlib | 3.11.2 | 3.11.2 |
| SciPy | not installed | 1.18.1 |
| Pinocchio (`pin`) | not installed | 4.1.0 |
| MathRobo | not installed | 0.0.4 |
| RoboKots | not installed | 0.0.1, revision below |
| Full suite | 886 passed, 35 skipped | 944 passed, no skips |
| `uv pip check` | passed (16 packages) | passed (37 packages) |

The core environment's skips were optional RoboKots, Pinocchio and JAX checks;
one skip was an entire module, so the two collected test counts differ.
The robotics environment imported the actual Pinocchio module and RoboKots
`_rust_core.cpython-313-darwin.so`, and loaded `examples/models/planar2.urdf`.
RoboKots was installed from revision
`3703e91697c737d4ce1e91ef025c9f580ef45e9b`.

Commands used:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/rei-verify-py311 uv sync --locked --python 3.11 --group dev
UV_PROJECT_ENVIRONMENT=/tmp/rei-verify-native313 uv sync --locked --python 3.13 \
  --group dev --group kots --group pinocchio
MPLBACKEND=Agg /tmp/rei-verify-py311/bin/python -m pytest -q -rs
MPLBACKEND=Agg /tmp/rei-verify-native313/bin/python -m pytest -q -rs --fail-on-skip
uv pip check --python /tmp/rei-verify-py311/bin/python
uv pip check --python /tmp/rei-verify-native313/bin/python
uv lock --check --offline
```

The CI robotics subset also passed all 109 tests without skips. Skip-policy
regressions cover runtime skips, collection skips and ordinary non-strict use.
The constrained IOC example recovered `[.75, .25]`, multiplier `3`, and the
demonstration to floating-point precision; its numerical fitter agreed with
SciPy SLSQP on 100 random small problems (objective difference <= `4.44e-16`).

The GitHub Actions workflow was checked for YAML and shell syntax. Both Linux
dependency selections passed `uv sync --locked --python-platform
x86_64-unknown-linux-gnu --dry-run`. This is **not** a Linux execution result:
the remote workflow has not been run. Native cyipopt/liteopt execution and
dynamic constrained IOC reconstruction remain outside this verification.

## Verification on 2026-10-14 after remote integration

Integrated local `e13e542` and remote `2265c27`, retaining the solver/runtime/IOC
refactoring together with TOML model loading, motion-order inference and batched
numerical/autodiff derivatives. Both branches' commit histories are preserved.

The lock now selects RoboKots revision
`acb2a041453d7c4d3170050e98a1ead81c1897ba`; other locked package versions did
not change. The earlier tables remain records of their respective revisions.

| Check on macOS ARM64 | Result |
|---|---|
| Python 3.11.14, core + dev | 899 passed, 73 optional-dependency skips |
| Python 3.13.1, dev + real RoboKots Rust / Pinocchio | 1013 passed, no skips (`--fail-on-skip`) |
| `uv pip check`, both environments | passed |
| `uv lock --check --offline` | passed |
| Workflow YAML and shell syntax | passed |

The Python 3.13 environment was created for the actual merged workspace;
`rei.__file__` and RoboKots' installed Git revision were checked explicitly.
The existing Python 3.11 core environment was reused because its locked
dependencies did not change. Commands:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/rei-merge-verify313 uv sync --locked --python 3.13 \
  --group dev --group kots --group pinocchio
MPLBACKEND=Agg /tmp/rei-verify-py311/bin/python -m pytest -q -rs
MPLBACKEND=Agg /tmp/rei-merge-verify313/bin/python -m pytest -q -rs --fail-on-skip
uv pip check --python /tmp/rei-verify-py311/bin/python
uv pip check --python /tmp/rei-merge-verify313/bin/python
uv lock --check --offline
```

The new AD test module skips before importing the optional backend when RoboKots
is absent, so core CI can collect successfully. The native CI selection now
also includes AD batching, motion-order inference and TOML model loading tests.
GitHub Actions / Linux execution has not been performed locally.

## Dependency update verification on 2026-11-05

Updated the existing project environment on macOS 15.7.4 ARM64, Python 3.13.1,
using the committed lock with the dev, kots and pinocchio groups. RoboKots now
tracks `main`; the lock resolves to `24acfc36a54e889332db798004b37122ce77843d`.

| Dependency / check | Result |
|---|---|
| MathRobo | 0.0.6 |
| RoboKots / robokots-native | 0.0.1 / 0.0.1, native API 1 |
| NumPy / SciPy / JAX | 2.5.3 / 1.18.1 / 0.11.2 |
| Pinocchio / pytest | 4.1.0 / 9.1.1 |
| Full suite (`--fail-on-skip`) | 1013 passed, no skips |
| CI portable selection, native import deliberately blocked | 23 passed, 115 deselected, no skips |
| Trajectory dynamics example, NumPy and Rust | Both converged in 9 iterations, cost 0.4154 |
| `uv pip check` | Passed, 38 packages |

Installed the separately supplied local wheel
`robokots_native-0.0.1-cp313-cp313-macosx_11_0_arm64.whl` after synchronizing.
Its SHA-256 is `a6772f039cfb74b286f27d2c7aac6561740b2cfd93c91b988ec33f831d44bb2c`.
No native source build was required. See [testing instructions](../../docs/testing.md)
for preserving the additional wheel during subsequent uv runs.

The initial suite had two failures comparing batched and stepwise IOC residuals:
the maximum discrepancy was `1.023e-12` at a residual magnitude of approximately
121 (relative difference `8.48e-15`). CM coefficient reconstruction and fused
products change floating-point accumulation order. The comparison now allows
`64 * float64 epsilon` relative error in addition to the existing `1e-12`
absolute tolerance; the weight and cache/call-count assertions are unchanged.
Rei's calculation code required no API or derivative-unit changes.

The revised workflow runs the portable selection using RoboKots NumPy and
Pinocchio. CI does not install or execute the RoboKots Rust backend; native
coverage is verified locally with the separately supplied wheel.
GitHub Actions / Linux and Python 3.11 were not rerun for this dependency update.

### Follow-up Rei integration review, 2026-11-05

The project environment passes **1016 tests with no skips** using
`MPLBACKEND=Agg .venv/bin/python -m pytest -q -rs --fail-on-skip`.
The additional three tests run the NumPy trajectory example and verify its
actual history files for LM, Krylov line search and Krylov trust region.
Krylov defaults to line search; the example now exposes `--globalization`
and routes the appropriate history options.

The portable CI selection was expanded from 23 to **105 passing tests**:
99 tests in portable modules with Rust parameters excluded and 6 NumPy
variants in modules containing unlabelled native-only tests. Both selections
were run in fresh Python subprocesses with `robokots_native` imports blocked;
there were no skips. GitHub Actions / Linux itself was not run locally.

The noisy DOC/IOC example and the Pinocchio/NumPy comparison example both
run successfully after explicitly selecting dense assembly for their dense
forward solvers. The compiler API's default remains `mul`, with dense-solver
guidance added to its docstring and README. The uv documentation now distinguishes
optional `uv run --no-sync` from `uv sync --inexact`, which preserves extra packages.

## Python matrix script verification, 2026-11-05

The new `scripts/test-python.sh` was exercised on macOS 15.7.4 ARM64 with
uv 0.9.18 and the current committed dependency lock. Every version/suite used
a separate environment under `/tmp/rei-python-matrix`; the project `.venv`
was not synchronized or replaced.

| Python | Core suite | NumPy / Pinocchio suite | Dependency checks |
|---|---|---|---|
| 3.11.14 | 904 passed, 76 skipped | 105 passed, no skips | Passed |
| 3.12.12 | 904 passed, 76 skipped | 105 passed, no skips | Passed |
| 3.13.1 | 904 passed, 76 skipped | 105 passed, no skips | Passed |
| 3.14.3 | 904 passed, 76 skipped | 105 passed, no skips | Passed |

The core environments contain only core/dev dependencies, so optional-backend
skips are expected and are listed in the logs. The robotics suite runs the
two portable selections from CI (99 and 6 passing tests), enforcing
`--fail-on-skip`. These matrix runs do not verify Rust wheels or native external
solvers, and they do not establish support on other operating systems.

The script's own five regression tests also pass, including continuation
after sync/dependency-check/pytest failures, final nonzero exit status,
the robotics selection, and rejecting unsupported Python before invoking uv.
The script was verified with macOS Bash 3.2, including empty pytest arguments.

Reproduce:

```sh
./scripts/test-python.sh --suite all 3.11 3.12 3.13 3.14 \
  --env-dir /tmp/rei-python-matrix --log-dir /tmp/rei-python-matrix-results
```

The successful run saved individual logs and a machine-readable summary to
`/tmp/rei-python-matrix-results/20261105-231246-12373/summary.tsv`.
Future runs use a new run directory and may select a newer installed patch
version unless an exact version is requested.

Final pre-commit verification on 2026-11-06: the project Python 3.13.1
environment, including the installed Rust extension, passes **1021 tests with
no skips** (`--fail-on-skip`). `uv lock --check --offline` and `uv pip check`
also pass. The shell script passes `bash -n`, the CI YAML parses, and the saved
benchmark result files parse as JSON.
