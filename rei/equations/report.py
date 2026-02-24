from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ..core.outcome import SolveOutcome
from .stationarity import StationarityTermContribution, term_constraint_kind

Array = np.ndarray


def _as_index_list(indices: Sequence[int] | Iterable[int]) -> list[int]:
    return [int(i) for i in indices]


def _as_vector(
    x: Array | Sequence[float],
    *,
    name: str,
) -> Array:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector.")
    return arr


def _format_vector(
    x: Array | Sequence[float],
    *,
    precision: int = 6,
    max_elems: int = 64,
) -> str:
    arr = _as_vector(x, name="x")
    if arr.size == 0:
        return "[]"
    if arr.size <= int(max_elems):
        return np.array2string(arr, precision=int(precision), separator=", ", suppress_small=True)
    shown = arr[: int(max_elems)]
    s = np.array2string(shown, precision=int(precision), separator=", ", suppress_small=True)
    if s.endswith("]"):
        s = s[:-1] + ", ...]"
    else:
        s = s + " ..."
    return f"{s} (len={arr.size})"


def build_ioc_log_sections(
    *,
    active_mode: str,
    active_idx: Sequence[int] | Iterable[int],
    active_grad_idx: Sequence[int] | Iterable[int],
    active_res_idx: Sequence[int] | Iterable[int],
    term_indices: Sequence[int] | Iterable[int],
    w_hat: Array | Sequence[float],
    ioc_identifiable: bool | None = None,
    ikkt_ok: bool | None = None,
    ikkt_residual: float | None = None,
    ikkt_tol: float | None = None,
    ioc_max_iters: int | None = None,
    w_true: Array | Sequence[float] | None = None,
    kkt: Any | None = None,
    simplex_out: SolveOutcome | None = None,
    callback_rows: int | None = None,
    precision: int = 6,
) -> list[tuple[str, object]]:
    """Build IOC-focused extra sections for text logs."""

    active_idx_l = _as_index_list(active_idx)
    active_grad_idx_l = _as_index_list(active_grad_idx)
    active_res_idx_l = _as_index_list(active_res_idx)
    term_indices_l = _as_index_list(term_indices)
    w_hat_vec = _as_vector(w_hat, name="w_hat")

    result: dict[str, object] = {
        "active_idx_selected": active_idx_l,
        "active_idx_gradient": active_grad_idx_l,
        "active_idx_residual": active_res_idx_l,
        "term_indices": term_indices_l,
        "w_hat": _format_vector(w_hat_vec, precision=int(precision)),
    }
    if ioc_identifiable is not None:
        result["identifiable"] = bool(ioc_identifiable)
    if ikkt_ok is not None:
        result["ikkt_ok"] = bool(ikkt_ok)
    if ikkt_residual is not None:
        result["ikkt_residual_norm"] = float(ikkt_residual)

    if w_true is not None:
        w_true_vec = _as_vector(w_true, name="w_true")
        if w_true_vec.size != w_hat_vec.size:
            raise ValueError(
                f"build_ioc_log_sections: w_true size mismatch. "
                f"Expected {w_hat_vec.size}, got {w_true_vec.size}."
            )
        result["w_true"] = _format_vector(w_true_vec, precision=int(precision))
        result["w_l1_error"] = float(np.linalg.norm(w_hat_vec - w_true_vec, ord=1))

    settings: dict[str, object] = {"active_mode": str(active_mode)}
    if ioc_max_iters is not None:
        settings["ioc_max_iters"] = int(ioc_max_iters)
    if ikkt_tol is not None:
        settings["ikkt_tol"] = float(ikkt_tol)

    sections: list[tuple[str, object]] = [("ioc.settings", settings), ("ioc.result", result)]

    if callback_rows is not None:
        sections.insert(
            0,
            (
                "solve.iter_meta",
                {"callback_rows": int(callback_rows)},
            ),
        )

    if kkt is not None:
        sections.append(
            (
                "kkt",
                {
                    "ok": bool(getattr(kkt, "ok")),
                    "stationarity_inf": float(getattr(kkt, "stationarity_inf")),
                    "eq_violation_inf": float(getattr(kkt, "eq_violation_inf")),
                    "ineq_violation_inf": float(getattr(kkt, "ineq_violation_inf")),
                },
            )
        )

    if simplex_out is not None:
        simplex_stats = simplex_out.stats
        sections.append(
            (
                "simplex.result",
                {
                    "status": str(simplex_stats.status),
                    "converged": bool(simplex_stats.converged),
                    "iters": int(simplex_stats.iterations),
                    "objective": float(simplex_stats.objective or float("nan")),
                    "meta": dict(simplex_out.meta),
                    "timing_total_seconds": float(simplex_out.timing.total_seconds),
                },
            )
        )

    return sections


def format_ioc_report(
    *,
    active_mode: str,
    active_idx: Sequence[int] | Iterable[int],
    active_grad_idx: Sequence[int] | Iterable[int],
    active_res_idx: Sequence[int] | Iterable[int],
    term_indices: Sequence[int] | Iterable[int],
    w_hat: Array | Sequence[float],
    ioc_identifiable: bool | None = None,
    ikkt_ok: bool | None = None,
    ikkt_residual: float | None = None,
    ikkt_tol: float | None = None,
    w_true: Array | Sequence[float] | None = None,
    kkt: Any | None = None,
    simplex_out: SolveOutcome | None = None,
    contributions: Iterable[StationarityTermContribution] | None = None,
    include_stationarity_terms: bool = False,
    title: str = "IOC",
    precision: int = 6,
    max_elems: int = 64,
) -> str:
    """Format an IOC-focused human-readable report."""

    active_idx_l = _as_index_list(active_idx)
    active_grad_idx_l = _as_index_list(active_grad_idx)
    active_res_idx_l = _as_index_list(active_res_idx)
    term_indices_l = _as_index_list(term_indices)
    w_hat_vec = _as_vector(w_hat, name="w_hat")

    lines: list[str] = [f"{str(title)}:"]
    lines.append(f"- active_mode={str(active_mode)}")
    if ioc_identifiable is not None:
        lines.append(f"- identifiable={bool(ioc_identifiable)}")
    if ikkt_ok is not None or ikkt_residual is not None or ikkt_tol is not None:
        ok_text = "n/a" if ikkt_ok is None else str(bool(ikkt_ok))
        res_text = "n/a" if ikkt_residual is None else f"{float(ikkt_residual):.3e}"
        tol_text = "n/a" if ikkt_tol is None else f"{float(ikkt_tol):.3e}"
        lines.append(f"- iKKT: ok={ok_text} residual_norm={res_text} tol={tol_text}")
        if ioc_identifiable is False:
            lines.append(
                "- iKKT note: no active objective stationarity terms were selected. "
                "This IOC setup is unidentifiable at the current forward solution."
            )

    lines.append(f"- active local idx (selected)={active_idx_l}")
    lines.append(f"- active local idx (gradient)={active_grad_idx_l}")
    lines.append(f"- active local idx (residual)={active_res_idx_l}")
    lines.append(f"- term_indices={term_indices_l}")
    lines.append(f"- w_hat={_format_vector(w_hat_vec, precision=int(precision), max_elems=int(max_elems))}")

    if w_true is not None:
        w_true_vec = _as_vector(w_true, name="w_true")
        if w_true_vec.size != w_hat_vec.size:
            raise ValueError(
                f"format_ioc_report: w_true size mismatch. "
                f"Expected {w_hat_vec.size}, got {w_true_vec.size}."
            )
        lines.append(
            f"- w_true={_format_vector(w_true_vec, precision=int(precision), max_elems=int(max_elems))}"
        )
        lines.append(f"- L1 error={float(np.linalg.norm(w_hat_vec - w_true_vec, ord=1)):.3e}")

    if kkt is not None:
        lines.append(
            "- KKT: "
            f"ok={bool(getattr(kkt, 'ok'))} "
            f"stationarity_inf={float(getattr(kkt, 'stationarity_inf')):.3e} "
            f"eq_violation_inf={float(getattr(kkt, 'eq_violation_inf')):.3e} "
            f"ineq_violation_inf={float(getattr(kkt, 'ineq_violation_inf')):.3e}"
        )

    if simplex_out is not None:
        simplex_stats = simplex_out.stats
        lines.append(
            "- simplex: "
            f"status={simplex_stats.status} "
            f"converged={simplex_stats.converged} "
            f"iters={simplex_stats.iterations} "
            f"objective={float(simplex_stats.objective or float('nan')):.3e}"
        )

    if include_stationarity_terms:
        if contributions is None:
            raise ValueError(
                "format_ioc_report: contributions is required when include_stationarity_terms=True."
            )
        contrib_list = list(contributions)
        lines.append("")
        lines.append("Stationarity terms:")
        for i, term in enumerate(contrib_list):
            is_constraint, kind = term_constraint_kind(dict(term.attrs))
            kind_str = "" if kind is None else f", kind={kind}"
            ttype = "constraint" if is_constraint else "objective"
            grad_norm = float(np.linalg.norm(np.asarray(term.gradient, dtype=float).reshape(-1)))
            w_hat_i = float(w_hat_vec[i]) if i < w_hat_vec.size else float("nan")
            lines.append(
                f"- [local={i}] term[{int(term.term_index)}] {term.name} ({ttype}{kind_str}) "
                f"||J^T r||={grad_norm:.3e} "
                f"||r_w||={float(term.weighted_residual_norm or 0.0):.3e} "
                f"w_hat={w_hat_i:.3e}"
            )

    return "\n".join(lines)


__all__ = [
    "build_ioc_log_sections",
    "format_ioc_report",
]
