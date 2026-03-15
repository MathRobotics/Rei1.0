from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..core.state_cache import StateKey
from ..term_gradient_matrix import build_term_gradient_matrix_from_terms
from ..xops import set_runtime_x

Array = np.ndarray


@dataclass(frozen=True)
class StationarityTermContribution:
    """Per-term contribution vector g_i = J_i^T r_i and diagnostics."""

    term_index: int
    name: str
    attrs: Mapping[str, Any]
    gradient: Array
    cost_name: str | None = None
    residual_size: int | None = None
    residual_norm: float | None = None
    weighted_residual_norm: float | None = None
    reference_weight: float | None = None


class StationaritySource(Protocol):
    """Capability surface required for stationarity-side composition."""

    @property
    def n_total(self) -> int: ...

    def set_point(self, x: Array | Any) -> None: ...

    def required_list(self, required: Iterable[StateKey] | None = None) -> list[StateKey]: ...

    def term_contributions(
        self,
        *,
        required: Iterable[StateKey] | None = None,
    ) -> list[StationarityTermContribution]: ...


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
        "RuntimeStationaritySource: runtime must expose `.problem` "
        "or `.full_runtime.problem`."
    )


def _optional_runtime_problem(runtime: Any) -> Any | None:
    try:
        return _resolve_runtime_problem(runtime)
    except Exception:
        return None


def _scalar_weight_from_cost(cost: Any) -> float | None:
    w = getattr(cost, "w", None)
    if w is None:
        return None
    w_arr = np.asarray(w, dtype=float).reshape(-1)
    if w_arr.size != 1:
        return None
    return float(w_arr[0])


def normalize_simplex_nonnegative(x: Array | Any) -> Array:
    """Normalize vector onto nonnegative simplex without solving optimization."""

    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size == 0:
        raise ValueError("normalize_simplex_nonnegative: cannot normalize empty vector.")
    v = np.maximum(v, 0.0)
    s = float(v.sum())
    if s <= 0.0:
        return np.full(v.shape, 1.0 / float(v.size), dtype=float)
    return v / s


def term_constraint_kind(attrs: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Read constraint kind metadata from term attrs."""

    kind_raw = attrs.get("constraint_kind", attrs.get("constraint_type", None))
    kind = None if kind_raw is None else str(kind_raw).strip().lower()
    if kind == "":
        kind = None
    is_constraint = bool(attrs.get("is_constraint", False) or kind is not None)
    return is_constraint, kind


def filter_stationarity_contributions(
    contributions: Iterable[StationarityTermContribution],
    *,
    include_constraints: bool = True,
) -> list[StationarityTermContribution]:
    """Filter contributions for stationarity composition."""

    out: list[StationarityTermContribution] = []
    include_constraints_b = bool(include_constraints)
    for term in contributions:
        is_constraint, _kind = term_constraint_kind(dict(term.attrs))
        if is_constraint and not include_constraints_b:
            continue
        out.append(term)
    return out


def build_stationarity_gradient_matrix(
    contributions: Iterable[StationarityTermContribution],
    *,
    n_total: int,
) -> tuple[Array, tuple[int, ...]]:
    """Build A=[g_0,...,g_n] where g_i = J_i^T r_i (columns)."""

    terms = list(contributions)
    n_total_i = int(n_total)
    if n_total_i < 0:
        raise ValueError(
            f"build_stationarity_gradient_matrix: n_total must be >= 0, got {n_total_i}."
        )
    if len(terms) == 0:
        return np.zeros((n_total_i, 0), dtype=float), tuple()

    A_col = np.zeros((n_total_i, len(terms)), dtype=float)
    term_indices: list[int] = []
    for j, term in enumerate(terms):
        g = np.asarray(term.gradient, dtype=float).reshape(-1)
        if g.size != n_total_i:
            raise ValueError(
                "build_stationarity_gradient_matrix: gradient size mismatch for "
                f"term[{int(term.term_index)}]. Expected {n_total_i}, got {g.size}."
            )
        A_col[:, j] = g
        term_indices.append(int(term.term_index))
    return A_col, tuple(term_indices)


def select_active_stationarity_indices(
    contributions: Iterable[StationarityTermContribution],
    *,
    mode: str = "residual",
    grad_tol: float = 1e-10,
    residual_tol: float = 1e-10,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Select active contribution indices (local index in provided list)."""

    grad_tol_f = float(grad_tol)
    if grad_tol_f < 0.0:
        raise ValueError(
            f"select_active_stationarity_indices: grad_tol must be >= 0, got {grad_tol_f}."
        )
    residual_tol_f = float(residual_tol)
    if residual_tol_f < 0.0:
        raise ValueError(
            "select_active_stationarity_indices: residual_tol must be >= 0, "
            f"got {residual_tol_f}."
        )

    mode_raw = str(mode).strip().lower()
    if mode_raw in ("gradient", "grad"):
        mode_norm = "gradient"
    elif mode_raw in ("residual", "res"):
        mode_norm = "residual"
    else:
        raise ValueError(
            "select_active_stationarity_indices: mode must be one of "
            "'gradient' or 'residual'. "
            f"Got {mode!r}."
        )

    terms = list(contributions)
    if mode_norm == "residual":
        missing_residual = [
            int(t.term_index) for t in terms if t.weighted_residual_norm is None
        ]
        if len(missing_residual) > 0:
            raise ValueError(
                "select_active_stationarity_indices: mode='residual' requires "
                "weighted_residual_norm for all contributions. "
                f"Missing on term indices: {missing_residual}."
            )

    active_gradient_idx = tuple(
        i
        for i, term in enumerate(terms)
        if float(np.linalg.norm(np.asarray(term.gradient, dtype=float).reshape(-1))) > grad_tol_f
    )
    active_residual_idx = tuple(
        i
        for i, term in enumerate(terms)
        if float(term.weighted_residual_norm) > residual_tol_f
    )
    active_idx = active_gradient_idx if mode_norm == "gradient" else active_residual_idx
    return active_idx, active_gradient_idx, active_residual_idx


def build_reference_simplex_init(
    contributions: Iterable[StationarityTermContribution],
    active_indices: Iterable[int],
) -> Array | None:
    """Build normalized reference simplex init over active local indices."""

    terms = list(contributions)
    active = tuple(int(i) for i in active_indices)
    if len(active) == 0:
        return None
    for i in active:
        if i < 0 or i >= len(terms):
            raise IndexError(
                f"build_reference_simplex_init: active index out of range: {i}. "
                f"Expected 0..{len(terms) - 1}."
            )

    refs: list[float] = []
    for i in active:
        ref = terms[i].reference_weight
        if ref is None:
            return None
        refs.append(float(ref))
    return normalize_simplex_nonnegative(np.asarray(refs, dtype=float))


@dataclass
class RuntimeStationaritySource:
    """Stationarity source adapter for runtimes exposing linearize_terms()."""

    runtime: Any

    @property
    def n_total(self) -> int:
        pack = getattr(self.runtime, "pack", None)
        if pack is None:
            raise AttributeError("RuntimeStationaritySource: runtime must expose `.pack`.")
        n_total = getattr(pack, "n_total", None)
        if n_total is None:
            raise AttributeError("RuntimeStationaritySource: runtime.pack must expose `.n_total`.")
        return int(n_total)

    def set_point(self, x: Array | Any) -> None:
        set_runtime_x(self.runtime, x, name="x")

    def required_list(self, required: Iterable[StateKey] | None = None) -> list[StateKey]:
        fn = getattr(self.runtime, "required_list", None)
        if not callable(fn):
            raise AttributeError(
                "RuntimeStationaritySource: runtime must expose callable required_list(required)."
            )
        return list(fn(required))

    def term_contributions(
        self,
        *,
        required: Iterable[StateKey] | None = None,
    ) -> list[StationarityTermContribution]:
        linearize_terms = getattr(self.runtime, "linearize_terms", None)
        if not callable(linearize_terms):
            raise AttributeError(
                "RuntimeStationaritySource: runtime must expose callable "
                "linearize_terms(required=..., weighted=...)."
            )

        req = self.required_list(required)
        terms_raw = list(linearize_terms(required=req, weighted=False))
        terms_weighted = list(linearize_terms(required=req, weighted=True))
        if len(terms_raw) != len(terms_weighted):
            raise RuntimeError(
                "RuntimeStationaritySource: internal term linearization mismatch "
                "between raw and weighted terms."
            )
        if len(terms_raw) == 0:
            return []

        A_col, _ = build_term_gradient_matrix_from_terms(terms_raw, n_total=self.n_total)
        problem = _optional_runtime_problem(self.runtime)

        out: list[StationarityTermContribution] = []
        for j, (term_raw, term_w) in enumerate(zip(terms_raw, terms_weighted)):
            if int(term_raw.term_index) != int(term_w.term_index):
                raise RuntimeError(
                    "RuntimeStationaritySource: term index mismatch between raw and "
                    "weighted linearizations."
                )

            reference_weight = None
            cost_name = None
            if problem is not None:
                try:
                    _expr, cost = problem.terms[int(term_raw.term_index)]
                    reference_weight = _scalar_weight_from_cost(cost)
                    cost_name = str(getattr(cost, "name", cost.__class__.__name__))
                except Exception:
                    reference_weight = None
                    cost_name = None

            r = np.asarray(term_raw.residual, dtype=float).reshape(-1)
            r_w = np.asarray(term_w.residual, dtype=float).reshape(-1)
            out.append(
                StationarityTermContribution(
                    term_index=int(term_raw.term_index),
                    name=str(term_raw.name),
                    attrs=dict(term_raw.attrs),
                    gradient=np.asarray(A_col[:, j], dtype=float).reshape(-1).copy(),
                    cost_name=cost_name,
                    residual_size=int(r.size),
                    residual_norm=float(np.linalg.norm(r)),
                    weighted_residual_norm=float(np.linalg.norm(r_w)),
                    reference_weight=reference_weight,
                )
            )
        return out


__all__ = [
    "StationarityTermContribution",
    "StationaritySource",
    "RuntimeStationaritySource",
    "normalize_simplex_nonnegative",
    "term_constraint_kind",
    "filter_stationarity_contributions",
    "build_stationarity_gradient_matrix",
    "select_active_stationarity_indices",
    "build_reference_simplex_init",
]
