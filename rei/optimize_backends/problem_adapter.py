from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ..core.mapping import mapping_as_dict
from ..optimize.builder import compile_nls_problem
from ..optimize.runtime import NLSRuntime


@dataclass(frozen=True)
class BackendCompileResult:
    runtime: NLSRuntime
    prepared: Any
    state_builder: Any


class ProblemBackendAdapter(Protocol):
    """Backend adapter protocol without trajectory-specific assumptions."""

    def prepare_dsl(self, dsl: Mapping[str, Any]) -> Any: ...

    def build_state_builder(
        self,
        *,
        model: Any,
        data: Any,
        prepared: Any,
    ) -> Any: ...

    def validate_runtime(
        self,
        *,
        runtime: NLSRuntime,
        state_builder: Any,
        prepared: Any,
    ) -> None: ...


def _extract_dsl_for_compile(prepared: Any) -> dict[str, Any]:
    if isinstance(prepared, Mapping):
        return mapping_as_dict(prepared, where="prepared")

    prepared_dsl = getattr(prepared, "dsl", None)
    if isinstance(prepared_dsl, Mapping):
        return mapping_as_dict(prepared_dsl, where="prepared.dsl")

    raise TypeError(
        "compile_problem_with_adapter: adapter.prepare_dsl(dsl) must return a mapping "
        "or an object exposing `.dsl` as mapping."
    )


def compile_problem_with_adapter(
    dsl: Mapping[str, Any],
    *,
    model: Any,
    data: Any,
    adapter: ProblemBackendAdapter,
) -> BackendCompileResult:
    prepared = adapter.prepare_dsl(dsl)
    prepared_dsl = _extract_dsl_for_compile(prepared)

    state_builder = adapter.build_state_builder(
        model=model,
        data=data,
        prepared=prepared,
    )
    build_state = getattr(state_builder, "build_state", None)
    if not callable(build_state):
        raise TypeError(
            "compile_problem_with_adapter: adapter.build_state_builder(...) must return "
            "object with callable `.build_state`."
        )

    runtime = compile_nls_problem(prepared_dsl, build_state=build_state)
    adapter.validate_runtime(
        runtime=runtime,
        state_builder=state_builder,
        prepared=prepared,
    )
    return BackendCompileResult(
        runtime=runtime,
        prepared=prepared,
        state_builder=state_builder,
    )


__all__ = [
    "BackendCompileResult",
    "ProblemBackendAdapter",
    "compile_problem_with_adapter",
]
