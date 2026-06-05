# Custom Robotics Backends

This guide describes the lightest way to connect a robotics library to Rei.
For most custom integrations, start with `RoboticsStateProvider` or
`TrajectoryRoboticsStateProvider` before writing a dedicated backend class.

## Backend Contract

Rei consumes backend state through:

```python
build_state(x_all, *, pack=None, time=None, required=None) -> dict[StateKey, Any]
```

`RoboticsStateProvider` builds this function from backend method names:

- `update_model(q, model, data)`: sync your backend model to the current `q`.
- `resolve_state_ref(key, model, data)`: return a backend-native reference for a `StateKey`.
- `value_handler(q, key, state_ref)`: return the requested state vector.
- `jac_handler(q, key, state_ref)`: return the requested Jacobian.

## Implementation Checklist

Before using a custom backend in an optimization problem, verify:

- `update_model(q, model, data)` updates all backend caches needed by every registered field.
- `resolve_state_ref(key, model, data)` resolves the same owner, frame, and field requested by `StateKey`.
- every `value_handler` returns numeric data that can be normalized to shape `(m,)`.
- every `jac_handler` returns a numeric 2D matrix.
- `jacobian_wrt` is set to the variable the Jacobian columns correspond to.
- trajectory providers receive a `trajectory_map` whose `q_at(p, k)` and `dqdp_at(k)` agree on dimensions.
- derivative maps used by dynamics have the same parameter dimension as `trajectory_map`.
- the provider accepts and returns every `StateKey` required by the DSL problem.

`update_model` is called once per `build_state()` evaluation for a single-step
provider, before state handlers are evaluated. `TrajectoryRoboticsStateProvider`
updates the backend once per requested time index `k`; if `update_motion_model`
is provided, it is called instead of `update_model` with the composed motion
vector for that step.

## Single-Step Provider

```python
import numpy as np

from rei.backends.state.robotics import RoboticsStateProvider
from rei.optimize.builder import compile_nls_problem


class MyBackendAdapter:
    def update(self, q, model, data):
        del data
        model.forward(q)

    def ref(self, key, model, data):
        del data
        return model.frame(key.owner.owner_name)

    def pos(self, q, key, frame):
        del q, key
        return np.asarray(frame.translation, dtype=float).reshape(3)

    def pos_jac(self, q, key, frame):
        del q, key
        return np.asarray(frame.linear_jacobian, dtype=float)


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

By default, the provider also registers `dtype="coord"`, `owner_type="total_joint"`,
`field="q"` and its Jacobian `q_J_q`.

## Binding Table

The recommended registration style is a state-key to method-name table. The
key format is:

```text
<dtype>.<owner_type>.<field>
<dtype>.<owner_type>.<field>.J_<var>
```

Example:

```python
provider = RoboticsStateProvider.from_binding_table(
    model=my_robot_model,
    data={},
    handler_owner=adapter,
    update_model="update",
    resolve_state_ref="ref",
    bindings={
        "kinematics.link.pos": "pos",
        "kinematics.link.pos.J_q": "pos_jac",
        "kinematics.link.rot": "rot",
        "kinematics.link.rot.J_q": "rot_jac",
        "dynamics.total_joint.torque": "torque",
        "dynamics.total_joint.torque.J_q": "torque_jac",
    },
)
```

This means:

- `"kinematics.link.pos": "pos"` registers `adapter.pos(...)` for `pos`.
- `"kinematics.link.pos.J_q": "pos_jac"` registers `adapter.pos_jac(...)` for `pos_J_q`.
- `"dynamics.total_joint.torque": "torque"` registers `adapter.torque(...)` for `torque`.

The same API accepts a small text table:

```python
provider = RoboticsStateProvider.from_binding_table(
    model=my_robot_model,
    data={},
    handler_owner=adapter,
    update_model="update",
    resolve_state_ref="ref",
    bindings="""
    kinematics.link.pos = pos
    kinematics.link.pos.J_q = pos_jac
    """,
)
```

Known owner types in this compact form are `link`, `joint`, and `total_joint`;
pass `owner_types=(...)` to support additional owner type names.

`RobotFieldBinding(...)` and `RobotFieldHandler(...)` are lower-level APIs for
cases where the compact name table is not expressive enough.

## Handler Shapes

With `validate_handler_shapes=True` (the default), callbacks are checked at the
provider boundary.

| Callback | Required return |
| --- | --- |
| `value_handler` | numeric vector, normalized to shape `(m,)` |
| `jac_handler` with `jacobian_wrt=q_var` | numeric matrix with shape `(m, q_dim)` |
| `jac_handler` with `jacobian_wrt="state"` | numeric matrix with `m` rows; columns are backend state or motion dimension |

Use `jacobian_wrt=q_var` for single-step providers when the handler returns
`d value / d q` directly. This is the default for `RoboticsStateProvider`.

Use `.J_state` in binding tables when a Jacobian is with respect to
backend state or motion and should be chained by a trajectory provider.
For `TrajectoryRoboticsStateProvider.from_binding_table(...)`, this is the
recommended form for backend-space Jacobians.

Use `jacobian_wrt=p_var` only when a trajectory handler already returns
parameter-space Jacobians, such as `d value / d p`. In that case Rei will not
chain the matrix again.

Validation errors include the failed `StateKey`, handler name, field name,
expected shape, and actual shape. For example, a bad `pos_J_q` handler reports
whether the row count disagrees with `pos` or whether the column count disagrees
with the current `q` dimension.

## Default State References

If `resolve_state_ref` is omitted, the provider passes a `RobotStateRef` to
handlers:

```python
RobotStateRef(
    k=0,
    owner_type="link",
    owner_name="ee",
    dtype="kinematics",
    field="pos",
    frame="world",
    rel_frame=None,
)
```

If your backend has native frame or joint handles, pass `resolve_state_ref` and
return whatever object your handlers need.

## Trajectory Provider

Use `TrajectoryRoboticsStateProvider` when the decision variable is a trajectory
parameter vector `p` and states are requested at time index `k`.

```python
from rei.backends.state.robotics import TrajectoryRoboticsStateProvider

provider = TrajectoryRoboticsStateProvider.from_binding_table(
    model=my_robot_model,
    data={},
    trajectory_map=trajectory_map,
    trajectory_derivative_maps=trajectory_derivative_maps,
    handler_owner=adapter,
    p_var="p",
    update_model="update",
    resolve_state_ref="ref",
    bindings={
        "kinematics.link.pos": "pos",
        "kinematics.link.pos.J_state": "pos_jac",
    },
)
```

The provider evaluates `q(k) = trajectory_map.q_at(p, k)` and chains Jacobians
to parameter-space Jacobians when DSL requests fields like `pos_J_p`.

For trajectory providers:

- `trajectory_map.q_at(p, k)` must return the configuration vector for step `k`.
- `trajectory_map.dqdp_at(k)` must return `d q(k) / d p`.
- `trajectory_derivative_maps={1: velocity_map, 2: acceleration_map}` is needed
  when dynamics handlers depend on velocity, acceleration, or higher derivatives.
- every derivative map must have the same parameter dimension as `trajectory_map`.
- dynamics handlers that return `d value / d motion` should use binding keys
  ending in `.J_state`.

## Motion Layouts

Some dynamics libraries need velocities and accelerations in addition to `q`.
`TrajectoryRoboticsStateProvider` can pass a motion vector to
`update_motion_model(q, motion, k, model, data)`.

| `motion_layout` | Motion vector |
| --- | --- |
| `"q"` | `q(k)` only |
| `"stacked"` | `[q, dq, ddq, ...]` by derivative block |
| `"interleaved"` | `[q0, dq0, ddq0, q1, dq1, ddq1, ...]` |

Example for stacked dynamics:

```python
def update_motion_robot(q, motion, k, model, data):
    del q, k, data
    model.import_motion(motion)


provider = TrajectoryRoboticsStateProvider.from_binding_table(
    model=my_robot_model,
    data={},
    trajectory_map=trajectory_map,
    trajectory_derivative_maps={1: velocity_map, 2: acceleration_map},
    handler_owner=adapter,
    p_var="p",
    bindings={
        "dynamics.total_joint.torque": "torque",
        "dynamics.total_joint.torque.J_state": "torque_motion_jacobian",
    },
    update_motion_model=update_motion_robot,
    motion_layout="stacked",
    derivative_orders=(0, 1, 2),
)
```

Here `torque_motion_jacobian` should return `d torque / d motion`. Rei chains it
to `d torque / d p`.

## Contract Tests

Custom backend packages can test their provider contract without compiling a
full optimization problem:

```python
from rei.backends.state.robotics import (
    assert_provider_contract,
    assert_trajectory_provider_contract,
)
from rei.core.state_schema import make_jac_key, make_key

key_pos = make_key(
    k=0,
    owner_type="link",
    owner_name="ee",
    dtype="kinematics",
    field="pos",
)
key_pos_j = make_jac_key(
    k=0,
    owner_type="link",
    owner_name="ee",
    dtype="kinematics",
    field="pos",
    var="q",
)

assert_provider_contract(
    provider,
    sample_q=[0.0, 0.0],
    expected_fields=[key_pos, key_pos_j],
    expected_shapes={key_pos: (3,), key_pos_j: (3, 2)},
)
```

For trajectory providers, pass the parameter vector and `p`-space keys:

```python
key_pos_j_p = make_jac_key(
    k=3,
    owner_type="link",
    owner_name="ee",
    dtype="kinematics",
    field="pos",
    var="p",
)

assert_trajectory_provider_contract(
    trajectory_provider,
    sample_p=[0.0, 0.0, 0.0, 0.0],
    expected_keys=[key_pos_j_p],
    expected_shapes={key_pos_j_p: (3, 4)},
)
```

These helpers check that the provider accepts the keys, returns all requested
keys, returns numeric arrays, and matches optional exact shapes. They are small
enough to copy into downstream backend test suites if importing test helpers
from Rei is not desirable.

## Common Failures

`update_model` is not called for a requested step:
the provider probably does not accept that `StateKey`, or the required key was
not passed through the DSL evaluation. Check `provider.accepts(key)` and the
`required` list.

`jac_handler` returns `(q_dim, m)` instead of `(m, q_dim)`:
transpose the backend Jacobian before returning it. Rei expects rows to match
the value vector.

`TrajectoryRoboticsStateProvider` chains a Jacobian twice:
use `.J_p` only when the handler already returns `d value / d p`. Use `.J_state`
when the handler returns `d value / d q` or `d value / d motion`.

Dynamics output has the right rows but wrong parameter columns:
check that `trajectory_derivative_maps` use the same parameter vector as
`trajectory_map`, and that `motion_layout` matches the order expected by your
backend.

Frame-specific states are numerically wrong but shapes pass:
inspect `resolve_state_ref`. Shape validation cannot detect a frame mismatch,
wrong owner name, or stale backend frame handle.

## When To Write A Dedicated Backend

Use the callback providers when:

- your backend can update from a numeric `q` or motion vector,
- state references can be resolved from `StateKey`,
- dense Jacobians are available or cheap enough.

Write a dedicated backend when:

- you need backend-specific sparse, JVP, or VJP fast paths,
- the backend has significant API fallback behavior,
- state queries need caching beyond `StateKey` reference caching,
- you need special shape or frame conversions.

Dedicated backends can still reuse `TrajectoryStateBuilderMixin`,
`robotics/motion.py`, `rei.backends.state.trajectory`, and `rei.backends.state.jacobian_ops`.

## Dedicated Backend Internals

When a backend needs dense Jacobians, JVP, or VJP fast paths, keep the probing
logic behind a small capability facade instead of spreading fallback calls
through the builder. RoboKots uses this pattern with
`RoboKotsJacobianOperator`:

- `dense(state_ref)` returns a dense backend Jacobian.
- `jvp(state_ref, cols)` returns `J @ cols`.
- `vjp(state_ref, rhs)` returns `J.T @ rhs`.

The shared protocols live in `rei.backends.state.jacobian_ops`:

```python
from rei.backends.state.jacobian_ops import (
    DenseJacobianProvider,
    JvpProvider,
    VjpProvider,
)
```

For optional backend dependencies, use the same error style as the built-in
backends:

```python
from rei.backends.optional import import_optional_backend

backend = import_optional_backend(
    "my_robotics_backend",
    backend_name="my_package.rei_backend",
    install_hint="uv sync --group my-backend",
)
```

This keeps import failures actionable for users who only installed Rei's core
dependencies.
