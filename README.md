# Rei

`rei` is a Python toolkit for building, linearizing, and solving numerical
optimization problems through capability-oriented APIs.

Hand-written problems are described with compact TOML spec files, which are
converted to Rei's lower-level DSL before compilation. Backend code connects
through a single `build_state()` function, so the optimization layer can stay
independent from robotics, vision, or other state providers.

Development priorities and acceptance criteria are recorded in the
[improvement plan](docs/improvement-plan.md) (Japanese).

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
uv run python examples/minimize_quadratic.py
uv run --group dev python -m pytest tests
```

Optional solver and backend groups:

```bash
uv sync --group plotting
uv sync --group pinocchio
uv sync --group kots
uv sync --group solver-liteopt
```

For editable pip installs outside the recommended `uv` workflow, use
`python -m pip install -e .`.

## Quick Start

```python
from rei import compile_nls_problem_spec_toml, solve

runtime = compile_nls_problem_spec_toml(
    "examples/spec/basic.toml",
    build_state=lambda *_args, **_kwargs: {},
)
out = solve(runtime)  # classical Levenberg–Marquardt

print(out.solution)
print(out.stats.status)
```

The same structure can also be built from a Python dict:

```python
from rei import compile_nls_problem_spec, solve

spec = {
    "opt_vals": {"joint_angles": {"dim": 2, "init": [0.0, 0.0]}},
    "terms": [
        {
            "name": "q_minus_target",
            "var": "joint_angles",
            "target": [0.5, -1.2],
        }
    ],
}
runtime = compile_nls_problem_spec(spec, build_state=lambda *_args, **_kwargs: {})
out = solve(runtime)
```

Backend state targets can also use a compact dotted state key:

```python
spec = {
    "opt_vals": {"joint_angles": {"dim": 7, "init": [0.0] * 7}},
    "terms": [
        {
            "name": "ee_pos",
            "state": "kinematics.link.ee.pos",
            "var": "joint_angles",
            "target": [0.4, 0.1, 0.3],
            "constraint": "eq",
        }
    ],
}
```

TOML spec は標準の人間向けテキスト入口です。低レベル DSL は
compile/debug/advanced 用の内部表現として残しています。

Trajectory terms can use reserved quantities instead of spelling out the
trajectory derivative map:

```toml
[[terms]]
name = "qdot_init"
kind = "eq"
weight = 100.0
quantity = "joint_velocities"
at = "first"
target = { fill = 0.0 }
```

The initial reserved quantities are `joint_angles`, `joint_velocities`,
`joint_accelerations`, and backend-computed `joint_torques`.

関節角 `q` の評価項は `trajectory` の有無に関係なく同じ形で書けます。
`trajectory` がある場合は軌道パラメータからの `q(k)`、ない場合は直接の
関節角変数 `q`（または top-level `joint.var`）に展開されます。

```toml
[[terms]]
name = "q_goal"
type = "joint_target"
at = "last"
target = [1.57, 0.0]
weight = 100.0
kind = "eq"
```

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

For custom robotics libraries, `RoboticsStateProvider` offers a callback-based
adapter so you do not need to write a full backend class:

```python
from rei.backends.state.robotics import RoboticsStateProvider


class MyBackendAdapter:
    def update(self, q, model, data):
        model.forward(q)

    def ref(self, key, model, data):
        return model.frame(key.owner.owner_name)

    def pos(self, q, key, frame):
        del q, key
        return frame.translation

    def pos_jac(self, q, key, frame):
        del q, key
        return frame.linear_jacobian


adapter = MyBackendAdapter()

provider = RoboticsStateProvider.from_binding_table(
    model=my_robot_model,
    data={},
    handler_owner=adapter,
    update_model="update",
    resolve_state_ref="ref",
    bindings={
        "kinematics.link.pos": "pos",
        "kinematics.link.pos.J_q": "pos_jac",
    },
)

runtime = compile_nls_problem(dsl, build_state=provider.build_state)
```

Provider callbacks are normalized to numeric arrays. With the default
`validate_handler_shapes=True`, value callbacks must return non-empty vectors
and Jacobian callbacks must return 2D arrays with rows matching the value size.

For trajectory-parameterized problems, use `TrajectoryRoboticsStateProvider`.
It evaluates `q(k)` from trajectory parameters and chains backend Jacobians into
parameter-space Jacobians. See [Custom Robotics Backends](docs/custom-robotics-backend.md)
for the full callback contract, shape rules, and trajectory examples.

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

### IOC weight interpretation

`estimate_ioc_weights()` returns `weights` in the original, unweighted
objective scale, normalized to sum to one when active terms exist.
`scaled_weights` contains the internal coefficients after column scaling.
Check `validation` and `identifiability`: fitting objective stationarity does
not account for constraint KKT multipliers, and inactive or dependent terms
can prevent weight identification. See the [IOC validation report](docs/ioc-validation.md)
for known-answer tests, migration details, and remaining limitations.

### RoboKots Jacobians

Choose the differentiation method independently of the matrix strategy:

```python
compiled = compile_kots_trajectory_problem(
    problem, model=kots, jacobian_method="autodiff",
)
```

`jacobian_method` accepts `"analytic"` (default), `"numerical"` (RoboKots finite
differences), or `"autodiff"` (RoboKots JAX forward-mode AD). The same option is
available on both Kots state builders and `compile_trajectory_ioc_problem`
with `backend="kots"`; it applies to DOC Jacobians and IOC gradient products.
The trajectory example also accepts `--jacobian-method`.

AD requires JAX and supports rigid-body momentum, force, torque, and their time
derivatives for fixed/revolute/prismatic models; it does not support kinematic
outputs or kinetic energy. Numerical kinetic-energy derivatives are also
unsupported by RoboKots. Unsupported requests raise errors, without switching
to analytic derivatives. AD runs without JIT, in a local float64 context, and
forms a dense Jacobian before multiplying. Numerical/AD methods currently use
per-time-step evaluation even when `batch_trajectory=True`; analytic mode keeps
the batched fast paths. These alternatives are primarily useful for validation
and can be much slower than the default.

The Kots trajectory backend uses RoboKots multiply APIs by default for
trajectory-parameter dynamics Jacobians:

- `jacobian_mul(list[StateType], rhs)` for `J @ rhs`
- `jacobian_transpose_mul(list[StateType], rhs)` for `J.T @ rhs`

RoboKots owns its state internally. `KotsStateBuilder`,
`KotsTrajectoryStateBuilder`, and `compile_kots_trajectory_problem` accept
`data=None` (the default); do not access the removed `kots.state_dict_` attribute.

Dense Jacobian assembly is still available by passing
`jacobian_strategy="dense"` to `compile_kots_trajectory_problem`; otherwise the
default strategy is `"mul"`. The older `prefer_matvec_jacobian` option is kept
only as a deprecated compatibility alias.

For trajectory problems containing `total_joint` coordinates and dynamics,
Rei batches all requested time steps into RoboKots' batch state and Jacobian
APIs by default. Set `batch_trajectory=False` to retain per-step evaluation;
kinematic state requests automatically use that compatible per-step path.
IOC terms that `vstack` compatible trajectory dynamics states also batch their
`J_p.T @ rhs` evaluations through the same API, while mixed state stacks retain
the per-step fallback.
For an unchanged optimization point and time grid, the materialized RoboKots
batch outward state is shared by value, JVP, and IOC VJP evaluation. The cache
is invalidated when `p`, the time grid, gravity, model order, or batched motion
sequence changes.

For sequential windows with identical DSL structure and time grid, use
`compile_kots_trajectory_problem_template(...)` once, then update only the
window inputs. This retains the RoboKots model/adapter, trajectory maps, and
runtime structure:

```python
from rei.optimize_backends.kots import compile_kots_trajectory_problem_template

template = compile_kots_trajectory_problem_template(problem, model=kots)
for p_window, target_window in windows:
    template.update_window(p=p_window, constants={"window_target": target_window})
    result = estimate_ioc_weights(template)
```

Changing the time grid, B-spline shape, or term structure requires a new
template. Absolute-time sliding windows may also replace all maps explicitly;
the q map and the complete existing derivative-order set must have matching
dimensions:

```python
template.update_window(
    p=p_window,
    trajectory_map=map_q,
    trajectory_derivative_maps={1: map_qdot, 2: map_qddot, 3: map_qdddot},
    dt=1.0 / 120.0,
)
```

### RoboKots model perturbations

Pass `perturbation=` to sample model parameter uncertainty through RoboKots.
It accepts a `robokots.PerturbationSpec`, a configuration dictionary, or a
TOML file path. Install the pinned version with `uv sync --group kots`.

```python
compiled = compile_kots_trajectory_problem(
    problem,
    model=kots,
    perturbation={
        "seed": 42,
        "mass_relative_std": 0.05,
        "link_length_relative_std": 0.02,
        "link_cog_translation_std": 0.001,
    },
)
report = compiled.perturbation_report.to_dict()
perturbed_model = compiled.state_builder.model
```

The original model is unchanged. Parameters are sampled once per builder and
stay fixed during residual, Jacobian, JVP/VJP, and solver evaluations. This is
model uncertainty; it does not add fresh measurement noise to each evaluation.
Relative mass/length standard deviations above are in log space; translation
standard deviations are in metres. RoboKots also supports explicit `rules`
with parameter scopes and noise distributions; Rei passes their configuration
through to RoboKots for validation.

The same argument is accepted by `KotsStateBuilder`,
`KotsTrajectoryStateBuilder`, `compile_kots_trajectory_problem_template`, and
`compile_trajectory_ioc_problem(..., backend="kots", data=None)`.
Builders expose `perturbation_report`; IOC results expose it through
`compiled.compiled.perturbation_report`. Templates keep the sampled model
across `update_window()` calls. For a new sample, construct a new builder or
compile again from the nominal model with a different seed. Passing an already
perturbed model together with `perturbation=` applies another perturbation.
With `perturbation=None` (the default), the original model is used and the report
is `None`. A TOML path is resolved relative to the process working directory;
its root fields are the same as the dictionary above.

Run the [model uncertainty example](examples/robokots_model_perturbation.py):

```bash
uv run --group kots python examples/robokots_model_perturbation.py
uv run --group kots python examples/robokots_model_perturbation.py --perturbation examples/perturbation/planar2.toml --backend rust
```

### Observation noise between DOC and IOC

`prepare_noisy_ioc_trajectory` adds independent, zero-mean Gaussian noise to
the joint coordinates at each trajectory sample, then fits the observations
back to the same trajectory map. For B-splines this uses the scalar basis
without materializing the full Kronecker matrix.

```python
from rei import prepare_noisy_ioc_trajectory, estimate_ioc_weights

# doc and ioc use the same variable layout and trajectory maps.
# For a nullspace-reduced DOC solve, first lift the solution:
# full_point = reduction.lift(doc_outcome.solution)
full_point = doc_outcome.solution.copy()
start, stop = doc.runtime.pack.slices[doc.p_var]
observation = prepare_noisy_ioc_trajectory(
    doc.trajectory_map,
    full_point[start:stop],
    std=0.001,  # radians for revolute joints; metres for prismatic joints
    seed=42,
)
ioc_point = full_point.copy()
ioc_point[start:stop] = observation.p
result = estimate_ioc_weights(ioc, p=ioc_point)
```

`std` may also be a vector of length `q_dim` for per-joint noise levels.
The helper preserves the input parameters, trajectory map and global random
state. The same seed reproduces the same noise. `std=0` preserves the original
parameters exactly. The result contains `q_clean`, `noise`, `q_observed`,
`q_fitted`, fitted parameters `p`, `fit_rank` and `fit_residual_norm`.
Arrays of joint observations have shape `(steps, q_dim)`.

Noise is applied once, before IOC. IOC velocities and accelerations come from
the fitted parameters and existing derivative maps. Least-squares fitting
projects the observations into the trajectory space, so the fitted trajectory
generally differs from the raw noisy observations. The fit does not enforce
boundary conditions or joint limits, and endpoints also receive noise.
Rank-deficient fits retain the original parameters' unobserved component.

Run the complete DOC → observations → fit → clean/noisy IOC comparison:

```bash
uv run --group kots python examples/robokots_doc_noise_ioc.py --noise-std 0.001 --seed 42 --output /tmp/doc-noise-ioc.npz
```

The saved NPZ contains all three trajectories, sampled noise, fitted parameters,
noise settings and estimated weights. This example uses a constrained DOC;
the current IOC estimates objective stationarity without constraint multipliers.
Its diagnostics are printed, and its weights are not a claim of exact recovery
of the generating weights. Model perturbations (`perturbation=`) and observation
noise are separate settings and may be combined in an experiment.

### Joint Mechanical Power

`joint_power` evaluates the scalar mechanical power `torque.T @ velocity`.
Use a torque state expression and a trajectory velocity expression at the same
time index; its VJP propagates analytically to both inputs:

```python
power = {
    "type": "joint_power",
    "torque": {"type": "get_state", "key": torque_key, "jac": {"var": "p"}},
    "velocity": {
        "type": "get_traj_var",
        "var": "p",
        "derivative_order": 1,
        "k": k,
    },
}
```

For a non-negative effort objective, use `joint_power_squared` (the alias
`joint_power_sq` is also accepted).  It evaluates `(torque.T @ velocity)**2`
and propagates its analytic product-rule VJP to both torque and velocity:

```python
power_squared = {
    "type": "joint_power_squared",
    "torque": {"type": "get_state", "key": torque_key, "jac": {"var": "p"}},
    "qdot": {
        "type": "get_traj_var",
        "var": "p",
        "derivative_order": 1,
        "k": k,
    },
}
```

RoboKots kinetic energy is available as the canonical scalar
`StateType("total_body", "total_body", "kinetic_energy")`.  Register it
through `dynamics_fields=("kinetic_energy",)` (or let the trajectory compiler
infer it from the DSL), then use the normal `get_state` form.  Rei delegates
the value, JVP, and VJP through that StateType and chains its q/qdot derivative
to trajectory parameters:

```python
energy = {
    "type": "get_state",
    "key": {
        "k": k,
        "owner_type": "total_body",
        "owner_name": "total_body",
        "dtype": "dynamics",
        "field": "kinetic_energy",
    },
    "jac": {"var": "p"},
}
```

World-frame gravity can be forwarded to RoboKots dynamics with, for example,
`gravity=(0.0, 0.0, -9.81)`. Omitting it preserves RoboKots' backward-compatible
zero-gravity default:

```python
compiled = compile_kots_trajectory_problem(
    problem,
    model=kots,
    kots_backend="rust",
    gravity=(0.0, 0.0, -9.81),
)
```

For `total_joint` dynamics, Rei expands the request to a list of per-joint
`StateType("joint", joint_name, field)` entries and passes that list to
RoboKots. These references are prepared during trajectory compilation and
cached per builder, field, and effective frame, so time steps and repeated IOC
evaluations reuse them. Additional requests are cached on first use. Create a
new builder if the robot's joint names, order, or topology change.
Rei expects RoboKots to support list inputs for `jacobian`,
`jacobian_mul`, and `jacobian_transpose_mul`. If a list call is unavailable,
Rei falls back to per-joint calls where possible.

To compare the dense and multiply paths with a local RoboKots checkout:

```bash
PYTHONPATH=/path/to/RoboKots:. python developer/benchmarks/robokots_jacobian_mul.py --dofs 7 32
```

## Solvers

`solve()` now defaults to classical Levenberg–Marquardt (`"levenberg_marquardt"`).
The previous adaptive Gauss–Newton algorithm remains available as
`solver="gauss_newton"` with the same options/initial point.
LM uses a gain ratio to accept/reject full steps and update damping; it does not
use line search or the legacy gradient-progress acceptance near roundoff.
See [the LM baseline](docs/levenberg-marquardt.md) for equations and migration.

Both built-in solvers print state history without an external callback;
set `options={"verbose": False}` to silence it. LM records accepted/rejected
trial details in `outcome.trial_history`, separately from `outcome.history`.
`options={"history_path": "history.jsonl"}` streams states to that file and
trials to `history.lm_trials.jsonl`; `trial_history_path` overrides the latter.
The previous solver retains `outcome.line_search_history` and
`line_search_history_path`. See [solver history](docs/solver-history.md).

### Residual VJP operators

For large trajectory parameter vectors, use `NLSRuntime.weighted_residual_vjp`
to compute `J_weighted.T @ rhs` without materializing the stacked dense
Jacobian.  `rhs` is ordered like `eval_stacked_terms(weighted=True)`.  The
more general `residual_vjp(rhs, weighted=False)` applies the raw residual
Jacobian instead.

```python
rhs = np.ones(runtime.eval_stacked_terms(weighted=True).shape)
gradient = runtime.weighted_residual_vjp(rhs)
```

Built-in costs propagate their residual weighting in reverse before Rei calls
the expression VJP.  Custom costs used with this operator must implement
`residual_vjp(r, rhs)` alongside `apply(r, blocks)`.

### Sparse B-spline trajectory maps

Prepared maps are shared by the trajectory state builder and `get_traj_var`
expressions. To reuse maps also used for IOC diagnostics, pass
`trajectory_maps=maps` to `compile_kots_trajectory_problem`,
`compile_pinocchio_trajectory_problem`, or `compile_trajectory_ioc_problem`:

```python
from rei import build_trajectory_maps_with_derivatives

maps = build_trajectory_maps_with_derivatives(
    problem["trajectory"],  # include q_dim (or pass default_q_dim)
    max_derivative_order=3,
    derivative_wrt="time",
    default_steps=problem["time"]["N"] + 1,
    default_dt=problem["time"]["dt"],
)
compiled = compile_kots_trajectory_problem(
    problem, model=kots, max_derivative_order=3, trajectory_maps=maps,
)
assert compiled.trajectory_map is maps[0]
assert compiled.trajectory_derivative_maps[3] is maps[3]
```

The API accepts a list indexed by derivative order or `{order: TrajectoryMap}`.
A nonempty external collection must include order 0. Missing orders are built
without replacing existing maps or re-evaluating their B-spline bases.
`prepare_trajectory_problem_dsl` and the `compile_nls_problem*` APIs also accept
`trajectory_maps`; `DslBuildEnv.seed_trajectory_maps` is the internal injection
point. With no external maps, backend compilation generates and shares one set.

Maps must match the DSL's knots, samples, parameter ordering, `derivative_wrt`,
and `dt`. Rei validates orders and dimensions, but does not recompute supplied
maps to verify their numeric contents. Treat shared maps as read-only. The
compile-local cache separates DSL content, effective dimensions, derivative
units and time step; time and parameter derivatives are not interchangeable.
`derivative_wrt` on the compile API describes supplied maps; it does not
override an expression's explicit/default derivative units. In particular,
raw `get_traj_var` defaults to `"u"`; use `derivative_wrt="time"` for time
derivatives. Finite-difference paths also retain existing map objects.

B-spline derivative requests must satisfy `derivative_order <= degree`
(likewise `max_derivative_order <= degree`). Larger orders raise `ValueError`
instead of returning zero derivatives, including the nonuniform-sample path.

B-spline `TrajectoryMap` instances retain a block-sparse operator instead of
materializing `kron(basis, I)` in normal runtime paths.  Use
`apply(p)`, `apply_at(k, p)`, `apply_transpose(rhs)`, and
`apply_transpose_at(k, rhs)` for coefficient-to-trajectory and adjoint
trajectory-to-coefficient operations.  q through higher derivative maps use
the same representation.  `TrajectoryMap.A` and `dqdp_at()` remain available
for compatibility, but requesting a dense Jacobian intentionally materializes
the corresponding rows.

B-spline basis evaluation is vectorized over samples and control points.
Map construction normalizes both knots and samples to the knot domain and
caches up to 64 derivative basis matrices, independent of joint count. An
affine change of the parameter axis preserves control-column order; derivative
scaling is restored by `(parameter_scale / knot_span)**order`. Returned maps
own their arrays, so modifying one map cannot corrupt the cache.
Cache keys match normalized grids exactly (without rounding). Translated
windows reuse a basis when those grids match; floating-point differences may
cause a safe cache miss. Moving samples over fixed global knots generally
changes the basis and is not treated as an equivalent local window.
Run `uv run python developer/benchmarks/bspline_maps.py` to compare cold and
cached map generation against the scalar recurrence.

`solve()` accepts these solver names:

- `"levenberg_marquardt"`: classical LM baseline (default)
- `"lm-ls"`: LM directions with Armijo line search; gradient-only convergence
- `"gauss_newton"`: previous adaptive Gauss-Newton solver (preserved)
- `"gauss_newton_operator"`: JVP/VJP-based Gauss-Newton with damped CGLS (NumPy only)
- `"gauss_newton_krylov"`: JVP/VJP-native trust-region GN with preconditioned truncated CG
- `"scipy_minimize"`: requires `scipy`
- `"cyipopt"`: requires `cyipopt`
- `"liteopt"`: requires `liteopt`

Example:

```python
from rei import solve

out = solve(
    runtime,
    solver="levenberg_marquardt",
    options={"max_iters": 200, "tol_grad": 1e-10, "tol_dx": 1e-12},
)

print(out.solution)
print(out.stats)
print(out.timing)
```

The Gauss–Newton variants serve different purposes:

| Solver | Step computation | Globalization | Details |
|---|---|---|---|
| `gauss_newton` | Dense Jacobian and linear solve | Adaptive damping and line search | Existing baseline |
| `gauss_newton_operator` | JVP/VJP with damped CGLS | Existing GN damping and line search | [Options and limits](docs/gauss-newton-operator.md) |
| `gauss_newton_krylov` | JVP/VJP with preconditioned truncated CG | Trust region and adaptive inner tolerance | [Options and limits](docs/gauss-newton-krylov.md) |

Both product-based variants support initial points, residual weighting, term
selection, nullspace reduction, callbacks and JSONL histories. Generic models
provide `eval`, `jvp`, `vjp` and point/state methods; no `linearize` is required.
Expression products fall back to local derivative blocks where necessary, so
not every expression is fully matrix-free.

The CGLS variant retains an exact damping-floor column scan. Krylov removes
that scan and uses affine-residual curvature for preconditioning when small
enough, otherwise a fixed number of VJP probes. Its affine preconditioner is a
size-limited dense matrix. Inner diagnostics are in `out.meta["inner_solves"]`.
Use `line_search_history_path` for CGLS and `trial_history_path` for Krylov.

RoboKots products support torque and its first/second time derivatives with
appropriate model orders. The [DOC benchmark index](developer/benchmarks/kots_doc.md)
separates measured solver performance, torque-derivative convergence limits,
and research-only prototypes. A prototype's timing does not describe the
production solver.

LM stops on a small gradient OR a small relative step, as in the reference
algorithm. `out.meta["reason"]` distinguishes these; `gradient_converged`
reports whether `max(abs(J.T @ r)) <= tol_grad`. Use `tol_dx=0` to disable
step-based convergence. A numerical stall or iteration limit remains possible.

`solver="lm-ls"` combines damped least-squares directions with backtracking.
Both LM solvers default to `tol_grad=1e-8`; `lm-ls` declares convergence only
when this gradient tolerance is met. See [LM with line search](docs/lm-ls.md)
for settings, stopping reasons, and history output.

The preserved Gauss-Newton solver and `nls()` check stationarity before treating
a small step as convergence. `tol_grad` (default `1e-10`) bounds
`max(abs(J.T @ r))`. A small step with neither a sufficiently small residual
nor gradient is reported as `stalled`. Gauss-Newton also restores the last
accepted point if evaluating a line-search trial raises an exception.

When passing an existing `NLSRuntimeLinearProblem` adapter to `solve()` or
`as_solver_problem()`, omitted `weighted` and `term_indices` settings inherit
the adapter's configuration. Explicit overrides create a new view without
modifying the original adapter. On raw runtimes, weighting defaults to true
and all terms are selected. Generic linearized problems cannot apply term
selection or disable weighting and reject those options.

The projected-gradient solver checks `||x - project(x - J.T @ r)||` at the
returned point independently of the chosen step size. Small steps that do
not satisfy this stationarity tolerance return `stalled`; the diagnostic is
available as `out.meta["projected_gradient_norm"]`.

`as_constraint_problem(existing_adapter)` similarly preserves omitted settings;
explicit `kind` or `weighted` overrides create a new view. Its `linearize()`
evaluates the constraints once and returns a consistent residual/Jacobian pair.
This adapter retains the penalty residual representation; signed inequality
margins for KKT are available through `runtime.linearize_inequality_constraints()`.

Gauss–Newton validates tolerances, damping, and iteration settings before
changing the initial point. Negative/nonfinite settings and fractional iteration
counts raise `ValueError`, even if the initial residual is already zero.
`max_iters=0` remains available for evaluation without taking a step.

For `scipy_minimize`, `cyipopt`, and `liteopt`, unknown top-level option keys are
forwarded to the backend. Options that belong to another solver are rejected.

To connect your own solver, adapt the runtime without using `solve()`:

```python
from rei import as_solver_problem

problem = as_solver_problem(runtime)

x0 = problem.x0
r = problem.residual(x0)
J = problem.jacobian(x0)
f = problem.objective(x0)
g = problem.gradient(x0)
```

## Examples

Run examples from the repository root:

```bash
uv run python examples/minimize_quadratic.py
uv run python examples/get_state_minimal.py
uv run python examples/toml_spec_problem.py
uv run python examples/stationarity_ioc.py
```

Backend examples:

```bash
uv sync --group pinocchio
uv run python examples/pinocchio_ik.py
uv run python examples/pinocchio_trajectory_dynamics.py

uv sync --group kots
uv run python examples/robokots_ik.py
uv run python examples/robokots_trajectory_dynamics.py
```

See `examples/README.md` for the full sample list and spec/DSL/model file guide.

## Development

Run the test suite:

```bash
uv run --group dev python -m pytest tests
```

Compile-check the package:

```bash
uv run python -m compileall -q rei
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Removed Import Paths

Legacy flat import paths have been removed. Use the canonical namespaces above.

- `rei.backends.state.template` -> `rei.backends.state.dispatch.template`
- `rei.backends.state.composite` -> `rei.backends.state.dispatch.composite`
- `rei.backends.state.spatial` -> `rei.backends.state.robotics.spatial`
- `rei.backends.state.kots` -> `rei.backends.state.robotics.kots`
- `rei.backends.state.pinocchio` -> `rei.backends.state.robotics.pinocchio`
- `rei.backends.state.vision_pinhole` -> `rei.backends.state.vision.pinhole`
