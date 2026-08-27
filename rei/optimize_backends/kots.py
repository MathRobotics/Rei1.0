from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..backends.state.robotics.kots import KotsTrajectoryStateBuilder
from ..core.expr.nodes import ConstantExpr, RepeatConstantExpr
from ..core.state_schema import DTYPE_DYNAMICS, torque_derivative_order
from ..core.trajectory import TrajectoryMap
from ..optimize.dsl.trajectory_compile import PreparedTrajectoryProblemDsl
from ..optimize.runtime import NLSRuntime
from ._state_field_utils import (
    canonicalize_unique_fields,
    required_base_fields_in_order_from_dsl,
    validate_runtime_field_coverage,
)
from .trajectory_adapter import compile_trajectory_problem_with_adapter
from .trajectory_diagnostics import (
    TrajectoryProblemDiagnostics,
    filter_unsupported_terms_from_dsl,
    inspect_trajectory_problem_backend,
    normalize_unsupported_policy,
)


@dataclass(frozen=True)
class KotsTrajectoryCompiledProblem:
    runtime: NLSRuntime
    trajectory_map: TrajectoryMap
    trajectory_derivative_maps: dict[int, TrajectoryMap]
    p_var: str
    dt: float
    model_order: int
    dynamics_fields: tuple[str, ...] = ()
    gravity: tuple[float, float, float] | None = None
    diagnostics: TrajectoryProblemDiagnostics | None = None
    state_builder: KotsTrajectoryStateBuilder | None = None


@dataclass
class KotsTrajectoryProblemTemplate:
    """Reusable Kots trajectory runtime for structurally identical windows.

    The template intentionally retains the same RoboKots model, adapter, state
    builder, trajectory maps, expression tree, and runtime.  Per-window calls
    may change the trajectory parameter, constant targets, and opaque backend
    data only; changing the time grid or expression structure requires a new
    template.
    """

    compiled: KotsTrajectoryCompiledProblem

    @property
    def backend(self) -> str:
        return "kots"

    @property
    def diagnostics(self) -> TrajectoryProblemDiagnostics | None:
        return self.compiled.diagnostics

    @property
    def runtime(self) -> NLSRuntime:
        return self.compiled.runtime

    @property
    def model(self) -> Any:
        builder = self.compiled.state_builder
        if builder is None:
            raise RuntimeError("Kots trajectory template has no state builder.")
        return builder.model

    def _constant_exprs_by_name(self) -> dict[str, list[ConstantExpr | RepeatConstantExpr]]:
        found: dict[str, list[ConstantExpr | RepeatConstantExpr]] = {}
        seen: set[int] = set()

        def visit(value: Any) -> None:
            value_id = id(value)
            if value_id in seen:
                return
            seen.add(value_id)
            if isinstance(value, (ConstantExpr, RepeatConstantExpr)):
                found.setdefault(str(value.name), []).append(value)
                return
            if isinstance(value, Mapping):
                for item in value.values():
                    visit(item)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
                return
            fields = getattr(value, "__dataclass_fields__", None)
            if fields is not None:
                for name in fields:
                    if name not in {"vars", "value"}:
                        visit(getattr(value, name))

        for expr, _cost in self.runtime.problem.terms:
            visit(expr)
        return found

    def update_window(
        self,
        *,
        p: Any | None = None,
        constants: Mapping[str, Any] | None = None,
        data: Any = None,
    ) -> KotsTrajectoryCompiledProblem:
        """Update mutable window inputs without recompiling the problem.

        ``constants`` maps compiled ``const`` expression names to their base
        vectors.  A name may refer to multiple constants, which are all updated
        after shape validation.  Passing ``data`` replaces only the builder's
        opaque data object; it never replaces the RoboKots model or adapter.
        """
        builder = self.compiled.state_builder
        if builder is None:
            raise RuntimeError("Kots trajectory template has no state builder.")

        if p is not None:
            p_vec = np.asarray(p, dtype=float).reshape(-1)
            if p_vec.size != self.compiled.trajectory_map.p_dim:
                raise ValueError(
                    "KotsTrajectoryProblemTemplate: p size mismatch. "
                    f"Expected {self.compiled.trajectory_map.p_dim}, got {p_vec.size}."
                )
            start, stop = self.runtime.pack.slices[self.compiled.p_var]
            current = self.runtime.pack.get()
            if not np.array_equal(current[start:stop], p_vec):
                dx = np.zeros_like(current)
                dx[start:stop] = p_vec - current[start:stop]
                self.runtime.pack.apply_dx(dx)

        if constants is not None:
            available = self._constant_exprs_by_name()
            for name_raw, value_raw in constants.items():
                name = str(name_raw)
                targets = available.get(name, ())
                if len(targets) == 0:
                    known = ", ".join(sorted(available))
                    raise KeyError(
                        f"KotsTrajectoryProblemTemplate: no constant named {name!r}. "
                        f"Known constants: {known}."
                    )
                value = np.asarray(value_raw, dtype=float)
                for target in targets:
                    if value.shape != np.asarray(target.value).shape:
                        raise ValueError(
                            f"KotsTrajectoryProblemTemplate: constant {name!r} shape mismatch. "
                            f"Expected {np.asarray(target.value).shape}, got {value.shape}."
                        )
                    target.value = value.copy()

        if data is not None:
            builder.data = data

        # Constant/data changes are outside VariablePack revisions.  Always
        # clear state values; the Kots batch cache independently reuses only a
        # matching p/time/gravity/model-order outward state.
        self.runtime.state.invalidate()
        return self.compiled


def _infer_model_dof(model: Any) -> int | None:
    dof_fn = getattr(model, "dof", None)
    if callable(dof_fn):
        try:
            return int(dof_fn())
        except Exception:
            return None
    robot = getattr(model, "robot_", None)
    if robot is not None and hasattr(robot, "dof"):
        try:
            return int(getattr(robot, "dof"))
        except Exception:
            return None
    return None


def _infer_model_order(model: Any) -> int:
    order_fn = getattr(model, "order", None)
    if callable(order_fn):
        try:
            return max(1, int(order_fn()))
        except Exception:
            pass

    order_attr = getattr(model, "order_", None)
    if order_attr is None:
        return 1
    try:
        return max(1, int(order_attr))
    except Exception:
        return 1


def _canonicalize_dynamics_fields(
    dynamics_fields: Sequence[str] | None,
) -> tuple[str, ...] | None:
    return canonicalize_unique_fields(
        dynamics_fields,
        where="compile_kots_trajectory_problem",
        param_name="dynamics_fields",
    )


def _validate_model_order_for_dynamics_fields(
    *,
    model_order: int,
    dynamics_fields: Sequence[str] | None,
) -> None:
    if dynamics_fields is None:
        return
    for field in dynamics_fields:
        deriv_order = torque_derivative_order(str(field))
        if deriv_order is None or deriv_order <= 0:
            continue
        required_model_order = int(deriv_order) + 3
        if int(model_order) >= required_model_order:
            continue
        raise ValueError(
            "compile_kots_trajectory_problem: dynamics field "
            f"{field!r} requires RoboKots model order >= {required_model_order}. "
            f"Current model order is {int(model_order)}."
        )


def _validate_kots_runtime_dynamics_coverage(
    *,
    runtime: NLSRuntime,
    builder: KotsTrajectoryStateBuilder,
    dynamics_owner_type: str,
) -> None:
    validate_runtime_field_coverage(
        runtime=runtime,
        builder=builder,
        dtype=DTYPE_DYNAMICS,
        owner_type=dynamics_owner_type,
        error_prefix="compile_kots_trajectory_problem",
        builder_name="KotsTrajectoryStateBuilder",
        missing_hint=(
            "Add missing entries to `dynamics_fields` "
            "(e.g. include 'torque_d1' for first torque derivative), "
            "or remove corresponding get_state dynamics terms."
        ),
    )


@dataclass
class _KotsTrajectoryCompileAdapter:
    fields: Sequence[str] | None = None
    dynamics_fields: Sequence[str] | None = None
    dynamics_owner_type: str = "total_joint"
    prefer_matvec_jacobian: bool = False
    jacobian_strategy: str | None = None
    kots_backend: str | None = None
    gravity: Sequence[float] | None = None
    batch_trajectory: bool = True
    resolved_dynamics_fields: tuple[str, ...] = ()
    resolved_gravity: tuple[float, float, float] | None = None

    def infer_model_dof(self, model: Any) -> int | None:
        return _infer_model_dof(model)

    def infer_model_order(self, model: Any) -> int:
        return _infer_model_order(model)

    def _resolve_dynamics_fields(
        self,
        *,
        model: Any,
        data: Any,
        prepared: PreparedTrajectoryProblemDsl,
    ) -> tuple[str, ...] | None:
        del model, data
        dynamics_fields_use = _canonicalize_dynamics_fields(self.dynamics_fields)
        if dynamics_fields_use is None:
            requested_fields_order, unsupported_owner_types = required_base_fields_in_order_from_dsl(
                dsl=prepared.dsl,
                dtype=DTYPE_DYNAMICS,
                owner_type=self.dynamics_owner_type,
            )
            if unsupported_owner_types:
                unsupported = ", ".join(sorted(unsupported_owner_types))
                raise ValueError(
                    "compile_kots_trajectory_problem: DSL contains dynamics keys with unsupported owner_type(s): "
                    f"{unsupported}. Supported owner_type is {self.dynamics_owner_type!r}."
                )
            dynamics_fields_use = tuple(requested_fields_order) if len(requested_fields_order) > 0 else None

        _validate_model_order_for_dynamics_fields(
            model_order=int(prepared.model_order),
            dynamics_fields=dynamics_fields_use,
        )
        return dynamics_fields_use

    def build_state_builder(
        self,
        *,
        model: Any,
        data: Any,
        prepared: PreparedTrajectoryProblemDsl,
    ) -> KotsTrajectoryStateBuilder:
        dynamics_fields_use = self._resolve_dynamics_fields(
            model=model,
            data=data,
            prepared=prepared,
        )
        self.resolved_dynamics_fields = (
            tuple() if dynamics_fields_use is None else tuple(dynamics_fields_use)
        )
        builder = KotsTrajectoryStateBuilder(
            model,
            data,
            trajectory_map=prepared.trajectory_map,
            trajectory_derivative_maps=prepared.trajectory_derivative_maps,
            p_var=prepared.p_var,
            fields=self.fields,
            dynamics_fields=dynamics_fields_use,
            dynamics_owner_type=self.dynamics_owner_type,
            prefer_matvec_jacobian=self.prefer_matvec_jacobian,
            jacobian_strategy=self.jacobian_strategy,
            kots_backend=self.kots_backend,
            gravity=self.gravity,
            batch_trajectory=self.batch_trajectory,
        )
        self.resolved_gravity = builder.gravity
        return builder

    def validate_runtime(
        self,
        *,
        runtime: NLSRuntime,
        state_builder: KotsTrajectoryStateBuilder,
        prepared: PreparedTrajectoryProblemDsl,
    ) -> None:
        del prepared
        _validate_kots_runtime_dynamics_coverage(
            runtime=runtime,
            builder=state_builder,
            dynamics_owner_type=self.dynamics_owner_type,
        )


def compile_kots_trajectory_problem(
    dsl: Mapping[str, Any],
    *,
    model: Any,
    data: Any,
    p_var: str | None = None,
    max_derivative_order: int | None = None,
    derivative_wrt: str = "time",
    default_steps: int | None = None,
    default_q_dim: int | None = None,
    default_dt: float | None = None,
    fields: Sequence[str] | None = None,
    dynamics_fields: Sequence[str] | None = None,
    dynamics_owner_type: str = "total_joint",
    unsupported: str = "error",
    prefer_matvec_jacobian: bool = False,
    jacobian_strategy: str | None = None,
    kots_backend: str | None = None,
    gravity: Sequence[float] | None = None,
    batch_trajectory: bool = True,
) -> KotsTrajectoryCompiledProblem:
    model_order = _infer_model_order(model)
    max_derivative_order_use = max(0, model_order - 1) if max_derivative_order is None else int(max_derivative_order)
    unsupported_policy = normalize_unsupported_policy(unsupported)
    diagnostics = inspect_trajectory_problem_backend(
        dsl,
        backend="kots",
        model_order=model_order,
        max_derivative_order=max_derivative_order_use,
        dynamics_owner_type=dynamics_owner_type,
        unsupported_action=("skipped" if unsupported_policy == "warn_skip" else "error"),
    )
    if unsupported_policy == "error":
        dsl_use: Mapping[str, Any] = dsl
    else:
        dsl_use = filter_unsupported_terms_from_dsl(dsl, diagnostics)

    adapter = _KotsTrajectoryCompileAdapter(
        fields=fields,
        dynamics_fields=dynamics_fields,
        dynamics_owner_type=dynamics_owner_type,
        prefer_matvec_jacobian=prefer_matvec_jacobian,
        jacobian_strategy=jacobian_strategy,
        kots_backend=kots_backend,
        gravity=gravity,
        batch_trajectory=batch_trajectory,
    )
    compiled = compile_trajectory_problem_with_adapter(
        dsl_use,
        model=model,
        data=data,
        adapter=adapter,
        p_var=p_var,
        max_derivative_order=max_derivative_order,
        derivative_wrt=derivative_wrt,
        default_steps=default_steps,
        default_q_dim=default_q_dim,
        default_dt=default_dt,
    )

    return KotsTrajectoryCompiledProblem(
        runtime=compiled.runtime,
        trajectory_map=compiled.prepared.trajectory_map,
        trajectory_derivative_maps=compiled.prepared.trajectory_derivative_maps,
        p_var=compiled.prepared.p_var,
        dt=float(compiled.prepared.dt),
        model_order=int(compiled.prepared.model_order),
        dynamics_fields=tuple(adapter.resolved_dynamics_fields),
        gravity=adapter.resolved_gravity,
        diagnostics=diagnostics,
        state_builder=compiled.state_builder,
    )


def compile_kots_trajectory_problem_template(
    dsl: Mapping[str, Any],
    **kwargs: Any,
) -> KotsTrajectoryProblemTemplate:
    """Compile one reusable Kots trajectory template for sequential windows.

    Arguments are identical to :func:`compile_kots_trajectory_problem`.  Call
    :meth:`KotsTrajectoryProblemTemplate.update_window` before each IOC
    estimate instead of compiling the same problem structure again.
    """
    return KotsTrajectoryProblemTemplate(compiled=compile_kots_trajectory_problem(dsl, **kwargs))


__all__ = [
    "KotsTrajectoryCompiledProblem",
    "KotsTrajectoryProblemTemplate",
    "compile_kots_trajectory_problem",
    "compile_kots_trajectory_problem_template",
]
