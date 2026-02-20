from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..backends.state.vision.provider import CameraCalibrationStateProvider, ModelUpdateFn, VisionFieldHandler
from ..core.state_schema import DTYPE_VISION
from ..optimize.dsl import PreparedVisionCalibrationDsl, prepare_vision_calibration_problem_dsl
from ..optimize.runtime import NLSRuntime
from ._state_field_utils import (
    canonicalize_unique_fields,
    required_base_fields_in_order_from_dsl,
    validate_runtime_field_coverage,
)
from .problem_adapter import compile_problem_with_adapter


@dataclass(frozen=True)
class VisionCalibrationCompiledProblem:
    runtime: NLSRuntime
    p_var: str
    owner_type: str
    owner_name: str
    field: str
    k: int
    term_name: str
    n_observations: int
    fields: tuple[str, ...]


def _canonicalize_vision_fields(fields: Sequence[str] | None) -> tuple[str, ...] | None:
    return canonicalize_unique_fields(
        fields,
        where="compile_camera_calibration_problem",
        param_name="fields",
    )


def _validate_vision_runtime_field_coverage(
    *,
    runtime: NLSRuntime,
    builder: CameraCalibrationStateProvider,
    owner_type: str,
) -> None:
    validate_runtime_field_coverage(
        runtime=runtime,
        builder=builder,
        dtype=DTYPE_VISION,
        owner_type=owner_type,
        error_prefix="compile_camera_calibration_problem",
        builder_name="CameraCalibrationStateProvider",
        missing_hint=(
            "Add missing entries to `field_handlers` (or `fields`) "
            "or remove corresponding get_state vision terms."
        ),
    )


@dataclass
class _VisionCompileAdapter:
    p_var: str | None = None
    owner_type: str | None = None
    owner_name: str | None = None
    field: str | None = None
    k: int | None = None
    term_name: str | None = None
    observations: Sequence[float] | np.ndarray | None = None
    standardize_terms: bool = True
    fields: Sequence[str] | None = None
    field_handlers: Mapping[str, VisionFieldHandler] | None = None
    update_model: ModelUpdateFn | None = None
    allow_nonzero_k: bool = True
    resolved_fields: tuple[str, ...] = ()
    prepared: PreparedVisionCalibrationDsl | None = None

    def prepare_dsl(self, dsl: Mapping[str, Any]) -> PreparedVisionCalibrationDsl:
        prepared = prepare_vision_calibration_problem_dsl(
            dsl,
            p_var=self.p_var,
            owner_type=self.owner_type,
            owner_name=self.owner_name,
            field=self.field,
            k=self.k,
            term_name=self.term_name,
            observations=self.observations,
            standardize_terms=self.standardize_terms,
        )
        self.prepared = prepared
        return prepared

    def _resolve_fields(self, *, prepared: PreparedVisionCalibrationDsl) -> tuple[str, ...]:
        fields_use = _canonicalize_vision_fields(self.fields)
        if fields_use is None:
            requested_fields_order, unsupported_owner_types = required_base_fields_in_order_from_dsl(
                dsl=prepared.dsl,
                dtype=DTYPE_VISION,
                owner_type=prepared.owner_type,
            )
            if unsupported_owner_types:
                unsupported = ", ".join(sorted(unsupported_owner_types))
                raise ValueError(
                    "compile_camera_calibration_problem: DSL contains vision keys with unsupported owner_type(s): "
                    f"{unsupported}. Supported owner_type is {prepared.owner_type!r}."
                )
            fields_use = tuple(requested_fields_order)

        if len(fields_use) == 0:
            raise ValueError(
                "compile_camera_calibration_problem: no vision fields resolved from DSL. "
                "Set `fields` explicitly or add get_state(dtype='vision') terms."
            )
        return tuple(fields_use)

    def build_state_builder(
        self,
        *,
        model: Any,
        data: Any,
        prepared: PreparedVisionCalibrationDsl,
    ) -> CameraCalibrationStateProvider:
        fields_use = self._resolve_fields(prepared=prepared)
        self.resolved_fields = tuple(fields_use)
        return CameraCalibrationStateProvider(
            model=model,
            data=data,
            param_var=prepared.p_var,
            owner_type=prepared.owner_type,
            fields=fields_use,
            field_handlers=self.field_handlers,
            update_model=self.update_model,
            allow_nonzero_k=self.allow_nonzero_k,
        )

    def validate_runtime(
        self,
        *,
        runtime: NLSRuntime,
        state_builder: CameraCalibrationStateProvider,
        prepared: PreparedVisionCalibrationDsl,
    ) -> None:
        _validate_vision_runtime_field_coverage(
            runtime=runtime,
            builder=state_builder,
            owner_type=prepared.owner_type,
        )


def compile_camera_calibration_problem(
    dsl: Mapping[str, Any],
    *,
    model: Any,
    data: Any,
    p_var: str | None = None,
    owner_type: str | None = None,
    owner_name: str | None = None,
    field: str | None = None,
    k: int | None = None,
    term_name: str | None = None,
    observations: Sequence[float] | np.ndarray | None = None,
    standardize_terms: bool = True,
    fields: Sequence[str] | None = None,
    field_handlers: Mapping[str, VisionFieldHandler] | None = None,
    update_model: ModelUpdateFn | None = None,
    allow_nonzero_k: bool = True,
) -> VisionCalibrationCompiledProblem:
    adapter = _VisionCompileAdapter(
        p_var=p_var,
        owner_type=owner_type,
        owner_name=owner_name,
        field=field,
        k=k,
        term_name=term_name,
        observations=observations,
        standardize_terms=standardize_terms,
        fields=fields,
        field_handlers=field_handlers,
        update_model=update_model,
        allow_nonzero_k=allow_nonzero_k,
    )
    compiled = compile_problem_with_adapter(
        dsl,
        model=model,
        data=data,
        adapter=adapter,
    )
    if adapter.prepared is None:
        raise RuntimeError("compile_camera_calibration_problem: internal error: missing prepared dsl.")
    prepared = adapter.prepared
    return VisionCalibrationCompiledProblem(
        runtime=compiled.runtime,
        p_var=prepared.p_var,
        owner_type=prepared.owner_type,
        owner_name=prepared.owner_name,
        field=prepared.field,
        k=int(prepared.k),
        term_name=prepared.term_name,
        n_observations=int(prepared.observations.size),
        fields=tuple(adapter.resolved_fields),
    )


__all__ = [
    "VisionCalibrationCompiledProblem",
    "compile_camera_calibration_problem",
]
