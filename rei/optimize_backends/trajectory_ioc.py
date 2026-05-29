from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..equations import (
    RuntimeStationaritySource,
    build_reference_simplex_init,
    build_stationarity_gradient_matrix,
    filter_stationarity_contributions,
    select_active_stationarity_indices,
    solve_simplex_min_norm,
)
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


def _scale_columns(
    A: Array,
    contributions: list[Any],
    *,
    mode: str,
    eps: float,
) -> tuple[Array, Array]:
    mode_name = str(mode).strip().lower()
    if mode_name == "none":
        scales = np.ones((A.shape[1],), dtype=float)
    elif mode_name == "gradient_norm":
        norms = np.linalg.norm(A, axis=0)
        scales = 1.0 / np.maximum(norms, float(eps))
    elif mode_name == "residual_norm":
        norms = np.asarray(
            [
                0.0 if c.residual_norm is None else float(c.residual_norm)
                for c in contributions
            ],
            dtype=float,
        )
        scales = 1.0 / np.maximum(norms, float(eps))
    elif mode_name == "weighted_residual_norm":
        norms = np.asarray(
            [
                0.0 if c.weighted_residual_norm is None else float(c.weighted_residual_norm)
                for c in contributions
            ],
            dtype=float,
        )
        scales = 1.0 / np.maximum(norms, float(eps))
    else:
        raise ValueError(
            "stationarity_scaling must be one of 'none', 'gradient_norm', "
            "'residual_norm', or 'weighted_residual_norm'."
        )
    return A * scales.reshape(1, -1), scales


def _contribution_json(term: Any, *, weight: float | None = None) -> dict[str, Any]:
    out = {
        "term_index": int(term.term_index),
        "name": str(term.name),
        "attrs": dict(term.attrs),
        "residual_size": None if term.residual_size is None else int(term.residual_size),
        "residual_norm": None if term.residual_norm is None else float(term.residual_norm),
        "weighted_residual_norm": None
        if term.weighted_residual_norm is None
        else float(term.weighted_residual_norm),
        "gradient_norm": float(np.linalg.norm(np.asarray(term.gradient, dtype=float).reshape(-1))),
    }
    if weight is not None:
        out["weight"] = float(weight)
    return out


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
    backend, compiled_obj, diagnostics = _as_compiled_ioc(compiled)
    runtime = compiled_obj.runtime
    source = RuntimeStationaritySource(runtime)
    if p is not None:
        source.set_point(np.asarray(p, dtype=float).reshape(-1))

    required = source.required_list(None)
    contributions_all = source.term_contributions(required=required)
    contributions = filter_stationarity_contributions(
        contributions_all,
        include_constraints=include_constraints,
    )
    A, term_indices = build_stationarity_gradient_matrix(
        contributions,
        n_total=int(source.n_total),
    )
    A_scaled, scales = _scale_columns(
        A,
        contributions,
        mode=stationarity_scaling,
        eps=float(stationarity_scale_eps),
    )
    active_idx, active_grad_idx, active_res_idx = select_active_stationarity_indices(
        contributions,
        mode=active_mode,
    )

    weights = np.zeros((len(contributions),), dtype=float)
    simplex_out = None
    ikkt_residual = np.zeros((A.shape[0],), dtype=float)
    ikkt_residual_scaled = np.zeros((A.shape[0],), dtype=float)
    if len(active_idx) > 0:
        active = np.asarray(active_idx, dtype=int)
        x0 = build_reference_simplex_init(contributions, active_idx)
        simplex_out = solve_simplex_min_norm(
            A_scaled[:, active],
            x0=x0,
            method=simplex_method,
            max_iters=int(simplex_max_iters),
            tol=float(simplex_tol),
        )
        weights[active] = np.asarray(simplex_out.solution, dtype=float).reshape(-1)
        ikkt_residual = A @ weights
        ikkt_residual_scaled = A_scaled @ weights

    return {
        "backend": str(backend),
        "requested_terms": []
        if diagnostics is None
        else [dict(t) for t in diagnostics.requested_terms],
        "active_terms": [
            _contribution_json(contributions[i], weight=float(weights[i]))
            for i in active_idx
        ],
        "terms": [
            _contribution_json(term, weight=float(weights[i]))
            for i, term in enumerate(contributions)
        ],
        "skipped_terms": []
        if diagnostics is None
        else [d.to_json_dict() for d in diagnostics.unsupported_terms],
        "warnings": []
        if diagnostics is None
        else [d.to_json_dict() for d in diagnostics.warnings],
        "stationarity_scaling": {
            "mode": str(stationarity_scaling),
            "eps": float(stationarity_scale_eps),
            "column_scale": [float(x) for x in scales],
            "active_column_scale": [float(scales[i]) for i in active_idx],
        },
        "stationarity": {
            "term_indices": [int(i) for i in term_indices],
            "active_indices": [int(i) for i in active_idx],
            "active_gradient_indices": [int(i) for i in active_grad_idx],
            "active_residual_indices": [int(i) for i in active_res_idx],
            "ikkt_residual": ikkt_residual.tolist(),
            "ikkt_residual_scaled": ikkt_residual_scaled.tolist(),
            "ikkt_residual_norm": float(np.linalg.norm(ikkt_residual)),
            "ikkt_residual_scaled_norm": float(np.linalg.norm(ikkt_residual_scaled)),
        },
        "simplex": None
        if simplex_out is None
        else {
            "method": str(simplex_out.meta.get("method", simplex_method)),
            "status": str(simplex_out.stats.status),
            "iterations": int(simplex_out.stats.iterations),
            "objective": None
            if simplex_out.stats.objective is None
            else float(simplex_out.stats.objective),
            "residual_norm": None
            if simplex_out.stats.residual_norm is None
            else float(simplex_out.stats.residual_norm),
            "meta": dict(simplex_out.meta),
        },
        "weights": [float(w) for w in weights],
    }


__all__ = [
    "TrajectoryIocCompiledProblem",
    "compile_trajectory_ioc_problem",
    "estimate_ioc_weights",
]
