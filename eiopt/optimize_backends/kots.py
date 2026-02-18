from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..backends.state.kots import KotsTrajectoryStateBuilder
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


@dataclass(frozen=True)
class KotsTrajectoryCompiledProblem:
    runtime: NLSRuntime
    trajectory_map: TrajectoryMap
    trajectory_derivative_maps: dict[int, TrajectoryMap]
    p_var: str
    dt: float
    model_order: int
    dynamics_fields: tuple[str, ...] = ()


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
    resolved_dynamics_fields: tuple[str, ...] = ()

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
        return KotsTrajectoryStateBuilder(
            model,
            data,
            trajectory_map=prepared.trajectory_map,
            trajectory_derivative_maps=prepared.trajectory_derivative_maps,
            p_var=prepared.p_var,
            fields=self.fields,
            dynamics_fields=dynamics_fields_use,
            dynamics_owner_type=self.dynamics_owner_type,
        )

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
) -> KotsTrajectoryCompiledProblem:
    adapter = _KotsTrajectoryCompileAdapter(
        fields=fields,
        dynamics_fields=dynamics_fields,
        dynamics_owner_type=dynamics_owner_type,
    )
    compiled = compile_trajectory_problem_with_adapter(
        dsl,
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
    )


__all__ = [
    "KotsTrajectoryCompiledProblem",
    "compile_kots_trajectory_problem",
]
