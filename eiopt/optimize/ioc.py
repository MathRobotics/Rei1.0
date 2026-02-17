from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.state_cache import StateKey
from ._xops import set_runtime_x
from .simplex_weight_solver import estimate_weights_simplex
from .term_gradient_matrix import build_term_gradient_matrix_from_terms

Array = np.ndarray


@dataclass(frozen=True)
class IocTermInfo:
    index: int
    name: str
    residual_size: int
    residual_norm: float
    weighted_residual_norm: float
    grad_norm: float
    doc_weight: float | None
    is_constraint: bool
    constraint_kind: str | None


@dataclass(frozen=True)
class IocPreparationResult:
    """Prepared IOC ingredients based on per-term gradients J_i^T r_i."""

    gradient_matrix: Array
    term_indices: tuple[int, ...]
    term_infos: tuple[IocTermInfo, ...]
    active_objective_indices: tuple[int, ...]
    active_gradient_objective_indices: tuple[int, ...]
    active_residual_objective_indices: tuple[int, ...]
    active_mode: str
    estimated_weights: Array
    doc_weights_normalized: Array | None
    doc_weights_normalized_residual_active: Array | None
    matrix_rank: int
    singular_values: Array
    solve_info: Mapping[str, Any] | None


def _resolve_runtime_problem(runtime: Any) -> Any:
    problem = getattr(runtime, "problem", None)
    if problem is not None:
        return problem
    full_runtime = getattr(runtime, "full_runtime", None)
    if full_runtime is not None:
        full_problem = getattr(full_runtime, "problem", None)
        if full_problem is not None:
            return full_problem
    raise AttributeError(
        "prepare_ioc_weights: runtime must expose `.problem` "
        "or `.full_runtime.problem`."
    )


def _scalar_weight_from_cost(cost: Any) -> float | None:
    w = getattr(cost, "w", None)
    if w is None:
        return None
    w_arr = np.asarray(w, dtype=float).reshape(-1)
    if w_arr.size != 1:
        return None
    return float(w_arr[0])


def _normalize_simplex_nonnegative(x: Array | Any) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size == 0:
        raise ValueError("_normalize_simplex_nonnegative: cannot normalize empty vector.")
    v = np.maximum(v, 0.0)
    s = float(v.sum())
    if s <= 0.0:
        return np.full(v.shape, 1.0 / float(v.size), dtype=float)
    return v / s


def _term_constraint_kind(attrs: Mapping[str, Any]) -> tuple[bool, str | None]:
    kind_raw = attrs.get("constraint_kind", attrs.get("constraint_type", None))
    kind = None if kind_raw is None else str(kind_raw).strip().lower()
    if kind == "":
        kind = None
    is_constraint = bool(attrs.get("is_constraint", False) or kind is not None)
    return is_constraint, kind


def prepare_ioc_weights(
    runtime: Any,
    *,
    x_opt: Array | Any,
    required: Iterable[StateKey] | None = None,
    active_mode: str = "residual",
    active_grad_tol: float = 1e-10,
    active_residual_tol: float = 1e-10,
    weight_tol: float = 1e-10,
    max_iters: int = 50000,
) -> IocPreparationResult:
    """Prepare IOC weights and diagnostics from one runtime state."""

    active_grad_tol_f = float(active_grad_tol)
    if active_grad_tol_f < 0.0:
        raise ValueError(
            f"prepare_ioc_weights: active_grad_tol must be >= 0, got {active_grad_tol_f}."
        )
    active_residual_tol_f = float(active_residual_tol)
    if active_residual_tol_f < 0.0:
        raise ValueError(
            "prepare_ioc_weights: active_residual_tol must be >= 0, "
            f"got {active_residual_tol_f}."
        )
    active_mode_raw = str(active_mode).strip().lower()
    if active_mode_raw in ("gradient", "grad"):
        active_mode_norm = "gradient"
    elif active_mode_raw in ("residual", "res"):
        active_mode_norm = "residual"
    else:
        raise ValueError(
            "prepare_ioc_weights: active_mode must be one of "
            "'gradient' or 'residual'. "
            f"Got {active_mode!r}."
        )
    weight_tol_f = float(weight_tol)
    if weight_tol_f <= 0.0:
        raise ValueError(f"prepare_ioc_weights: weight_tol must be > 0, got {weight_tol_f}.")
    max_iters_i = int(max_iters)
    if max_iters_i <= 0:
        raise ValueError(f"prepare_ioc_weights: max_iters must be > 0, got {max_iters_i}.")

    problem = _resolve_runtime_problem(runtime)
    set_runtime_x(runtime, x_opt, name="x")
    required_list = runtime.required_list(required)
    term_linear = runtime.linearize_terms(required=required_list, weighted=False)
    term_linear_weighted = runtime.linearize_terms(required=required_list, weighted=True)
    if len(term_linear) == 0:
        raise ValueError("prepare_ioc_weights: no terms are available to construct IOC matrix.")
    if len(term_linear) != len(term_linear_weighted):
        raise RuntimeError(
            "prepare_ioc_weights: internal term linearization mismatch between raw and weighted terms."
        )

    A_col, term_indices = build_term_gradient_matrix_from_terms(
        term_linear,
        n_total=int(runtime.pack.n_total),
    )
    A = np.asarray(A_col, dtype=float).T
    singular_values = np.linalg.svd(A, compute_uv=False)
    matrix_rank = int(np.linalg.matrix_rank(A))

    infos: list[IocTermInfo] = []
    for j, (term, term_w) in enumerate(zip(term_linear, term_linear_weighted)):
        if int(term.term_index) != int(term_w.term_index):
            raise RuntimeError(
                "prepare_ioc_weights: term index mismatch between raw and weighted linearizations."
            )
        attrs = dict(term.attrs)
        is_constraint, kind = _term_constraint_kind(attrs)
        _expr, cost = problem.terms[int(term.term_index)]
        grad = np.asarray(A_col[:, j], dtype=float).reshape(-1)
        r = np.asarray(term.residual, dtype=float).reshape(-1)
        r_w = np.asarray(term_w.residual, dtype=float).reshape(-1)
        infos.append(
            IocTermInfo(
                index=int(term.term_index),
                name=str(term.name),
                residual_size=int(r.size),
                residual_norm=float(np.linalg.norm(r)),
                weighted_residual_norm=float(np.linalg.norm(r_w)),
                grad_norm=float(np.linalg.norm(grad)),
                doc_weight=_scalar_weight_from_cost(cost),
                is_constraint=is_constraint,
                constraint_kind=kind,
            )
        )

    active_gradient_idx = tuple(
        i
        for i, info in enumerate(infos)
        if info.grad_norm > active_grad_tol_f
    )
    active_residual_idx = tuple(
        i
        for i, info in enumerate(infos)
        if info.weighted_residual_norm > active_residual_tol_f
    )
    active_idx = active_gradient_idx if active_mode_norm == "gradient" else active_residual_idx

    w_ioc = np.zeros((len(infos),), dtype=float)
    solve_info: Mapping[str, Any] | None = None
    doc_weight_all = [info.doc_weight for info in infos]
    doc_weights_normalized: Array | None = None
    doc_weights_normalized_residual_active: Array | None = None

    if len(active_idx) > 0:
        active_arr = np.asarray(active_idx, dtype=int)
        A_active_col = np.asarray(A_col[:, active_arr], dtype=float)
        x0_active: Array | None = None
        if all(w is not None for w in doc_weight_all):
            x0_active = _normalize_simplex_nonnegative(
                np.asarray([doc_weight_all[i] for i in active_idx], dtype=float)
            )

        w_active, solve_info_raw = estimate_weights_simplex(
            A_active_col,
            max_iters=max_iters_i,
            tol=weight_tol_f,
            x0=x0_active,
            return_info=True,
        )
        w_ioc[active_arr] = np.asarray(w_active, dtype=float).reshape(-1)
        solve_info = dict(solve_info_raw)

        if all(w is not None for w in doc_weight_all):
            doc_weights_normalized = np.zeros((len(infos),), dtype=float)
            doc_weights_normalized[active_arr] = _normalize_simplex_nonnegative(
                np.asarray([doc_weight_all[i] for i in active_idx], dtype=float)
            )

    if len(active_residual_idx) > 0 and all(w is not None for w in doc_weight_all):
        active_residual_arr = np.asarray(active_residual_idx, dtype=int)
        doc_weights_normalized_residual_active = np.zeros((len(infos),), dtype=float)
        doc_weights_normalized_residual_active[active_residual_arr] = _normalize_simplex_nonnegative(
            np.asarray([doc_weight_all[i] for i in active_residual_idx], dtype=float)
        )

    return IocPreparationResult(
        gradient_matrix=np.asarray(A_col, dtype=float).copy(),
        term_indices=tuple(int(i) for i in term_indices),
        term_infos=tuple(infos),
        active_objective_indices=active_idx,
        active_gradient_objective_indices=active_gradient_idx,
        active_residual_objective_indices=active_residual_idx,
        active_mode=active_mode_norm,
        estimated_weights=w_ioc,
        doc_weights_normalized=None if doc_weights_normalized is None else doc_weights_normalized.copy(),
        doc_weights_normalized_residual_active=(
            None
            if doc_weights_normalized_residual_active is None
            else doc_weights_normalized_residual_active.copy()
        ),
        matrix_rank=matrix_rank,
        singular_values=np.asarray(singular_values, dtype=float).copy(),
        solve_info=solve_info,
    )


def format_ioc_report(
    result: IocPreparationResult,
    *,
    include_singular_values: bool = False,
    include_term_details: bool = False,
    prefix: str = "[IOC prep]",
) -> str:
    """Format IOC preparation summary text."""

    pfx = str(prefix)
    lines: list[str] = []

    if len(result.active_objective_indices) == 0:
        lines.append(
            f"{pfx} no active terms for IOC estimation "
            f"(mode={result.active_mode})."
        )

    A_shape = tuple(np.asarray(result.gradient_matrix, dtype=float).T.shape)
    lines.append(
        f"{pfx} A shape={A_shape}, rank={int(result.matrix_rank)}, "
        f"active terms (IOC-active, mode={result.active_mode})="
        f"{list(result.active_objective_indices)}"
    )
    lines.append(
        f"{pfx} active terms (gradient-based)="
        f"{list(result.active_gradient_objective_indices)}"
    )
    lines.append(
        f"{pfx} active terms (residual-based)={list(result.active_residual_objective_indices)}"
    )
    if include_singular_values:
        lines.append(f"{pfx} singular values: {np.asarray(result.singular_values, dtype=float)}")

    if result.solve_info is None:
        lines.append(f"{pfx} estimated weights: {np.asarray(result.estimated_weights, dtype=float)}")
    else:
        lines.append(
            f"{pfx} estimated weights: {np.asarray(result.estimated_weights, dtype=float)} "
            "(solver=estimate_weights_simplex, "
            f"iters={int(result.solve_info.get('iterations', -1))}, "
            f"converged={bool(result.solve_info.get('converged', False))})"
        )

    if result.doc_weights_normalized is None:
        lines.append(f"{pfx} DOC scalar weights are not available for all terms.")
    else:
        w_doc = np.asarray(result.doc_weights_normalized, dtype=float)
        w_ioc = np.asarray(result.estimated_weights, dtype=float)
        lines.append(
            f"{pfx} normalized DOC weights (IOC-active terms, mode={result.active_mode}): {w_doc}"
        )
        lines.append(f"{pfx} ||w_doc - w_ioc||: {float(np.linalg.norm(w_doc - w_ioc))}")
    if result.doc_weights_normalized_residual_active is not None:
        lines.append(
            f"{pfx} normalized DOC weights (residual-active terms only): "
            f"{np.asarray(result.doc_weights_normalized_residual_active, dtype=float)}"
        )

    if set(result.active_gradient_objective_indices) != set(result.active_residual_objective_indices):
        lines.append(
            f"{pfx} note: residual-active terms can differ from gradient-active terms; "
            f"IOC estimation uses {result.active_mode}-active terms."
        )

    if include_term_details:
        for info, w_hat in zip(result.term_infos, np.asarray(result.estimated_weights, dtype=float)):
            w_doc = "n/a" if info.doc_weight is None else f"{info.doc_weight:.6e}"
            term_type = "constraint" if info.is_constraint else "objective"
            kind = "" if info.constraint_kind is None else f", kind={info.constraint_kind}"
            lines.append(
                f"{pfx} term[{info.index}] {info.name} ({term_type}{kind}): "
                f"m={info.residual_size}, ||r||={info.residual_norm:.6e}, ||r_w||={info.weighted_residual_norm:.6e}, "
                f"||J^T r||={info.grad_norm:.6e}, w_doc={w_doc}, w_ioc={float(w_hat):.6e}"
            )

    return "\n".join(lines)


__all__ = [
    "IocTermInfo",
    "IocPreparationResult",
    "prepare_ioc_weights",
    "format_ioc_report",
]
