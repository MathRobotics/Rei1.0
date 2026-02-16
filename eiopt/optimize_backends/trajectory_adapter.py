from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ..optimize.builder import compile_nls_problem
from ..optimize.dsl.trajectory_compile import PreparedTrajectoryProblemDsl, prepare_trajectory_problem_dsl
from ..optimize.runtime import NLSRuntime


@dataclass(frozen=True)
class BackendTrajectoryCompileResult:
    runtime: NLSRuntime
    prepared: PreparedTrajectoryProblemDsl
    state_builder: Any


class TrajectoryBackendAdapter(Protocol):
    def infer_model_dof(self, model: Any) -> int | None: ...

    def infer_model_order(self, model: Any) -> int: ...

    def build_state_builder(
        self,
        *,
        model: Any,
        data: Any,
        prepared: PreparedTrajectoryProblemDsl,
    ) -> Any: ...

    def validate_runtime(
        self,
        *,
        runtime: NLSRuntime,
        state_builder: Any,
        prepared: PreparedTrajectoryProblemDsl,
    ) -> None: ...


def compile_trajectory_problem_with_adapter(
    dsl: Mapping[str, Any],
    *,
    model: Any,
    data: Any,
    adapter: TrajectoryBackendAdapter,
    p_var: str | None = None,
    max_derivative_order: int | None = None,
    derivative_wrt: str = "time",
    default_steps: int | None = None,
    default_q_dim: int | None = None,
    default_dt: float | None = None,
) -> BackendTrajectoryCompileResult:
    model_dof = adapter.infer_model_dof(model)
    model_order = adapter.infer_model_order(model)

    prepared = prepare_trajectory_problem_dsl(
        dsl,
        p_var=p_var,
        model_dof=model_dof,
        model_order=model_order,
        max_derivative_order=max_derivative_order,
        derivative_wrt=derivative_wrt,
        default_steps=default_steps,
        default_q_dim=default_q_dim,
        default_dt=default_dt,
    )
    state_builder = adapter.build_state_builder(
        model=model,
        data=data,
        prepared=prepared,
    )
    runtime = compile_nls_problem(prepared.dsl, build_state=state_builder.build_state)
    adapter.validate_runtime(
        runtime=runtime,
        state_builder=state_builder,
        prepared=prepared,
    )
    return BackendTrajectoryCompileResult(
        runtime=runtime,
        prepared=prepared,
        state_builder=state_builder,
    )


__all__ = [
    "BackendTrajectoryCompileResult",
    "TrajectoryBackendAdapter",
    "compile_trajectory_problem_with_adapter",
]
