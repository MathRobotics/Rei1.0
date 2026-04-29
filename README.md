# Rei

`rei` is a Python toolkit for building, linearizing, and solving numerical
optimization problems through capability-oriented APIs.

Problems are described with Python dictionaries or TOML DSL files. Backend code
connects through a single `build_state()` function, so the optimization layer can
stay independent from robotics, vision, or other state providers.

## Requirements

- Python `>=3.11`
- Core dependency: `numpy`
- Optional plotting: `matplotlib`
- Optional solvers/backends are installed only when needed

## Installation

This repository assumes `uv` for local development.

```bash
uv sync
```

Run commands through the managed environment:

```bash
uv run python examples/01_minimize_quadratic.py
uv run python -m pytest tests
```

Optional solver and backend groups:

```bash
uv sync --group pinocchio
uv sync --group kots
uv sync --group solver-liteopt
```

For editable pip installs outside the recommended `uv` workflow, use
`python -m pip install -e .`.

## Quick Start

```python
from rei import compile_nls_problem, solve

dsl = {
    "variables": [{"name": "q", "dim": 2, "init": [0.0, 0.0]}],
    "terms": [
        {
            "expr": {
                "type": "sub",
                "a": {"type": "get_var", "var": "q"},
                "b": {"type": "const", "var": "q", "value": [3.0, -1.0]},
            },
            "cost": {"type": "l2"},
        }
    ],
}

runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
out = solve(runtime, solver="gauss_newton")

print(out.solution)
print(out.stats.status)
```

Loading the same kind of problem from TOML:

```python
from rei import compile_nls_problem, load_problem_toml, solve

dsl = load_problem_toml("examples/dsl/basic.toml")
runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
out = solve(runtime)
```

DSL の書き方をまとめたガイドは `docs/dsl.md` を参照してください。

## Canonical Namespace

- `rei.optimize`: optimization entry points such as `compile_nls_problem` and `solve`
- `rei.problem`: generic problem capabilities and runtime adapters
- `rei.equations`: equation-system capabilities, including stationarity sources
- `rei.flow`: constraint and projection capability adapters
- `rei.backends.state`: backend state builders
- `rei.optimize_backends`: backend-specific compile helpers

## Backend Contract

Backends connect to `rei` through `build_state()`:

```python
build_state(x_all, *, pack=None, time=None, required=None) -> dict[StateKey, Any]
```

Arguments:

- `x_all`: full decision-variable vector
- `pack`: `VariablePack`, when variable metadata is needed
- `time`: `TimeGrid`, when compiling trajectory problems
- `required`: requested `StateKey` values; `None` means all relevant state

Implementation expectations:

- Return deterministic values for the same input.
- Return a `dict[StateKey, Any]`.
- Honor `required` when it is provided.
- Return numeric arrays with shapes expected by the DSL expressions.

## Capability Adapters

Runtime-like objects can be viewed through generic capabilities:

```python
from rei import as_constraint_problem, as_linear_equation_problem, as_project_problem

eq_problem = as_linear_equation_problem(runtime)
constraint_problem = as_constraint_problem(runtime, kind="eq")
project_problem = as_project_problem(runtime)
```

For linearized least-squares problems:

```python
from rei import as_linearized_problem

linear_problem = as_linearized_problem(runtime, weighted=True)
r, J = linear_problem.linearize()
```

## Backend Compile Helpers

Backend-specific helpers live under `rei.optimize_backends`:

```python
from rei.optimize_backends.kots import compile_kots_trajectory_problem
from rei.optimize_backends.pinocchio import compile_pinocchio_trajectory_problem
from rei.optimize_backends.vision import compile_camera_calibration_problem
```

Each helper returns a compile result whose main entry point is
`compiled.runtime`. Some helpers also return backend-specific metadata such as
trajectory maps or prepared DSL data.

## Solvers

`solve()` accepts these solver names:

- `"gauss_newton"`: built-in Gauss-Newton solver
- `"scipy_minimize"`: requires `scipy`
- `"cyipopt"`: requires `cyipopt`
- `"liteopt"`: requires `liteopt`

Example:

```python
from rei import solve

out = solve(
    runtime,
    solver="gauss_newton",
    options={"max_iters": 50, "tol_r": 1e-10, "tol_dx": 1e-10},
)

print(out.solution)
print(out.stats)
print(out.timing)
```

For `scipy_minimize`, `cyipopt`, and `liteopt`, unknown top-level option keys are
forwarded to the backend. Options that belong to another solver are rejected.

## Examples

Run examples from the repository root:

```bash
uv run python examples/01_minimize_quadratic.py
uv run python examples/02_get_state_minimal.py
uv run python examples/03_toml_problem.py
uv run python examples/08_camera_calibration.py
uv run python examples/10_stationarity_ioc.py
```

Backend examples:

```bash
uv sync --group pinocchio
uv run python examples/04_pinocchio_ik.py
uv run python examples/06_pinocchio_trajectory_dynamics.py

uv sync --group kots
uv run python examples/05_robokots_ik.py
uv run python examples/07_robokots_trajectory_dynamics.py
uv run python examples/09_kots_vision_composite.py
uv run python examples/11_forward_then_inverse_ioc.py
```

See `examples/README.md` for the full sample list and DSL/model file guide.

## Development

Run the test suite:

```bash
uv run python -m pytest tests
```

Compile-check the package:

```bash
uv run python -m compileall -q rei
```

`uv run python -m pytest` also collects files under `examples/`; depending on
the local `liteopt` version, `examples/test_liteopt.py` may require updating or
skipping.

## Removed Import Paths

Legacy flat import paths have been removed. Use the canonical namespaces above.

- `rei.backends.state.template` -> `rei.backends.state.dispatch.template`
- `rei.backends.state.composite` -> `rei.backends.state.dispatch.composite`
- `rei.backends.state.spatial` -> `rei.backends.state.robotics.spatial`
- `rei.backends.state.kots` -> `rei.backends.state.robotics.kots`
- `rei.backends.state.pinocchio` -> `rei.backends.state.robotics.pinocchio`
- `rei.backends.state.vision_pinhole` -> `rei.backends.state.vision.pinhole`
