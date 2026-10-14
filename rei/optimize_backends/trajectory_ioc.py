from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..equations import RuntimeStationaritySource, estimate_stationarity_weights
from .trajectory_diagnostics import TrajectoryProblemDiagnostics

Array = np.ndarray


@dataclass(frozen=True)
class TrajectoryIocCompiledProblem:
    backend: str
    compiled: Any
    diagnostics: TrajectoryProblemDiagnostics | None = None

    @property
    def runtime(self) -> Any:
        return self.compiled.runtime

    @property
    def trajectory_map(self) -> Any:
        return self.compiled.trajectory_map

    @property
    def trajectory_derivative_maps(self) -> Any:
        return self.compiled.trajectory_derivative_maps

    @property
    def p_var(self) -> str:
        return str(self.compiled.p_var)


def compile_trajectory_ioc_problem(
    dsl: Mapping[str, Any],
    *,
    backend: str,
    model: Any,
    data: Any,
    unsupported: str = "warn_skip",
    **kwargs: Any,
) -> TrajectoryIocCompiledProblem:
    backend_name = str(backend).strip().lower()
    if backend_name in ("pinocchio", "pin"):
        from .pinocchio import compile_pinocchio_trajectory_problem

        compiled = compile_pinocchio_trajectory_problem(
            dsl,
            model=model,
            data=data,
            unsupported=unsupported,
            **kwargs,
        )
        return TrajectoryIocCompiledProblem(
            backend="pinocchio",
            compiled=compiled,
            diagnostics=getattr(compiled, "diagnostics", None),
        )
    if backend_name in ("kots", "robokots"):
        from .kots import compile_kots_trajectory_problem

        compiled = compile_kots_trajectory_problem(
            dsl,
            model=model,
            data=data,
            unsupported=unsupported,
            **kwargs,
        )
        return TrajectoryIocCompiledProblem(
            backend="kots",
            compiled=compiled,
            diagnostics=getattr(compiled, "diagnostics", None),
        )
    raise ValueError("compile_trajectory_ioc_problem: backend must be 'pinocchio' or 'kots'.")


def _as_compiled_ioc(compiled: Any) -> tuple[str, Any, TrajectoryProblemDiagnostics | None]:
    if isinstance(compiled, TrajectoryIocCompiledProblem):
        return compiled.backend, compiled.compiled, compiled.diagnostics
    backend = str(getattr(compiled, "backend", "") or compiled.__class__.__name__)
    return backend, compiled, getattr(compiled, "diagnostics", None)


def estimate_ioc_weights(
    compiled: Any,
    *,
    p: Array | Any | None = None,
    include_constraints: bool = False,
    active_mode: str = "gradient",
    stationarity_scaling: str = "gradient_norm",
    stationarity_scale_eps: float = 1e-12,
    simplex_method: str = "qr_nullspace",
    simplex_max_iters: int = 2000,
    simplex_tol: float = 1e-10,
) -> dict[str, Any]:
    """Adapt a compiled trajectory to the backend-independent IOC fitter.

    See :func:`rei.equations.estimate_stationarity_weights` for coefficient
    normalization and the limitations of objective-only stationarity.
    """
    backend, compiled_obj, diagnostics = _as_compiled_ioc(compiled)
    result = estimate_stationarity_weights(
        RuntimeStationaritySource(compiled_obj.runtime),
        p=p,
        include_constraints=include_constraints,
        active_mode=active_mode,
        stationarity_scaling=stationarity_scaling,
        stationarity_scale_eps=stationarity_scale_eps,
        simplex_method=simplex_method,
        simplex_max_iters=simplex_max_iters,
        simplex_tol=simplex_tol,
    )
    return {
        "backend": str(backend),
        "requested_terms": [] if diagnostics is None else [dict(t) for t in diagnostics.requested_terms],
        "skipped_terms": [] if diagnostics is None else [d.to_json_dict() for d in diagnostics.unsupported_terms],
        "warnings": [] if diagnostics is None else [d.to_json_dict() for d in diagnostics.warnings],
        **result,
    }


__all__ = [
    "TrajectoryIocCompiledProblem",
    "compile_trajectory_ioc_problem",
    "estimate_ioc_weights",
]
