from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.outcome import SolveOutcome
from ..core.state_cache import StateKey
from ..core.timing import TimingReport
from .kkt import KKTCheckResult, check_kkt_conditions
from .runtime import NLSRuntime

Array = np.ndarray

_DEFAULT_NAME_BLACKLIST = {
    "const",
    "const_repeat",
    "get_state",
    "get_var",
    "get_traj_var",
    "hinge",
    "stack",
    "sub",
    "time_diff",
}


@dataclass(frozen=True)
class NamedExprValue:
    term_index: int
    name: str
    expr_type: str
    value: Array


def _format_vector(x: Array, *, max_elems: int, precision: int) -> str:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return "[]"

    if x.size <= max_elems:
        return np.array2string(x, precision=precision, separator=", ", suppress_small=True)

    shown = x[:max_elems]
    s = np.array2string(shown, precision=precision, separator=", ", suppress_small=True)
    if s.endswith("]"):
        s = s[:-1] + ", ...]"
    else:
        s = s + " ..."
    return f"{s} (len={x.size})"


def _iter_expr_children(expr: Any) -> Iterator[Any]:
    a = getattr(expr, "a", None)
    if a is not None and hasattr(a, "eval"):
        yield a

    b = getattr(expr, "b", None)
    if b is not None and hasattr(b, "eval"):
        yield b

    base = getattr(expr, "base", None)
    if base is not None and hasattr(base, "eval"):
        yield base

    parts = getattr(expr, "parts", None)
    if isinstance(parts, (list, tuple)):
        for p in parts:
            if p is not None and hasattr(p, "eval"):
                yield p


def _walk_expr(expr: Any) -> Iterator[Any]:
    stack = [expr]
    while stack:
        cur = stack.pop()
        yield cur
        children = list(_iter_expr_children(cur))
        stack.extend(reversed(children))


def collect_named_expr_values(
    runtime: NLSRuntime,
    *,
    required: Iterable[StateKey] | None = None,
    name_blacklist: set[str] | None = None,
    include_blacklisted: bool = False,
) -> list[NamedExprValue]:
    """Collect evaluated values for named Expr nodes from all objective terms."""

    if name_blacklist is None:
        name_blacklist = set(_DEFAULT_NAME_BLACKLIST)

    problem = runtime.problem
    ctx = runtime.ctx
    required_list = runtime.required_list(required)
    runtime.update_state_if_needed(required=required_list)

    out: list[NamedExprValue] = []
    for i, (expr, _cost) in enumerate(problem.terms):
        for node in _walk_expr(expr):
            name = getattr(node, "name", None)
            if not isinstance(name, str) or not name:
                continue
            if (not include_blacklisted) and name in name_blacklist:
                continue

            y, _blocks = node.eval(ctx)
            out.append(
                NamedExprValue(
                    term_index=i,
                    name=name,
                    expr_type=node.__class__.__name__,
                    value=np.asarray(y, dtype=float).reshape(-1).copy(),
                )
            )
    return out


def get_named_expr_value(
    runtime: NLSRuntime,
    *,
    name: str,
    term_index: int | None = None,
    required: Iterable[StateKey] | None = None,
    name_blacklist: set[str] | None = None,
    include_blacklisted: bool = True,
) -> Array:
    """Get one named Expr value as a flat vector."""

    name = str(name)
    if name == "":
        raise ValueError("get_named_expr_value: name must be non-empty.")

    values = collect_named_expr_values(
        runtime,
        required=required,
        name_blacklist=name_blacklist,
        include_blacklisted=include_blacklisted,
    )
    matches = [v for v in values if v.name == name]
    if term_index is not None:
        matches = [v for v in matches if v.term_index == int(term_index)]

    if len(matches) == 0:
        where = f", term_index={int(term_index)}" if term_index is not None else ""
        raise ValueError(f"get_named_expr_value: no named Expr found for name={name!r}{where}.")

    if len(matches) > 1:
        locs = ", ".join(f"(t{v.term_index}, {v.expr_type})" for v in matches[:8])
        more = "" if len(matches) <= 8 else ", ..."
        raise ValueError(
            f"get_named_expr_value: multiple named Expr values matched name={name!r}. "
            f"Matches: {locs}{more}. "
            "Specify term_index to disambiguate."
        )

    return np.asarray(matches[0].value, dtype=float).reshape(-1).copy()


def _append_kkt_report(lines: list[str], out: KKTCheckResult) -> None:
    lines.append("")
    lines.append("KKT:")
    lines.append(f"- ok={out.ok} | message={out.message}")
    lines.append(
        f"- stationarity_inf={out.stationarity_inf:.3e} | "
        f"eq_violation_inf={out.eq_violation_inf:.3e} | "
        f"ineq_violation_inf={out.ineq_violation_inf:.3e}"
    )
    lines.append(
        f"- complementarity_inf={out.complementarity_inf:.3e} | "
        f"dual_violation_inf={out.dual_violation_inf:.3e} | "
        f"active_ineq_rows={int(out.n_active_ineq_rows)}/{int(out.n_ineq_rows)}"
    )


def _append_solve_summary(lines: list[str], summary: Mapping[str, Any]) -> None:
    parts: list[str] = []
    if "status" in summary:
        parts.append(f"status={str(summary['status'])}")
    if "converged" in summary:
        parts.append(f"converged={bool(summary['converged'])}")
    if "iters" in summary:
        parts.append(f"iters={int(summary['iters'])}")
    if "cost0" in summary:
        parts.append(f"cost0={float(summary['cost0']):.3e}")
    if "cost" in summary:
        parts.append(f"cost={float(summary['cost']):.3e}")
    if "rnorm" in summary:
        parts.append(f"rnorm={float(summary['rnorm']):.3e}")
    if "dxnorm" in summary:
        parts.append(f"dxnorm={float(summary['dxnorm']):.3e}")
    if "message" in summary and str(summary["message"]) != "":
        parts.append(f"message={str(summary['message'])}")
    if len(parts) == 0:
        return
    lines.append("Solve:")
    lines.append(f"- {' '.join(parts)}")
    lines.append("")


def _append_trajectory_summary(lines: list[str], summary: Mapping[str, Any]) -> None:
    parts: list[str] = []
    if "steps" in summary:
        parts.append(f"steps={int(summary['steps'])}")
    if "dt" in summary:
        parts.append(f"dt={float(summary['dt']):g}")
    if "p_dim" in summary:
        parts.append(f"p_dim={int(summary['p_dim'])}")
    if "dynamics_fields" in summary:
        parts.append(f"dynamics_fields={summary['dynamics_fields']}")
    if len(parts) == 0:
        return
    lines.append("Trajectory:")
    lines.append(f"- {' '.join(parts)}")
    lines.append("")


def format_timing_report(
    timing: TimingReport,
    *,
    title: str = "Timing",
    sort_by: str = "seconds",
    descending: bool = True,
    max_rows: int | None = None,
) -> str:
    """Format TimingReport as a compact ASCII table."""

    sort_key_raw = str(sort_by).strip().lower()
    if sort_key_raw in {"seconds", "time", "duration", ""}:
        sort_key = "seconds"
    elif sort_key_raw in {"name", "span"}:
        sort_key = "name"
    elif sort_key_raw in {"count", "calls"}:
        sort_key = "count"
    else:
        raise ValueError(
            "format_timing_report: sort_by must be one of "
            "'seconds', 'name', 'count'. "
            f"Got {sort_by!r}."
        )

    rows = [
        (
            str(span.name),
            float(span.seconds),
            int(span.count),
        )
        for span in timing.spans
    ]

    if sort_key == "seconds":
        rows.sort(key=lambda x: x[1], reverse=bool(descending))
    elif sort_key == "name":
        rows.sort(key=lambda x: x[0], reverse=bool(descending))
    else:
        rows.sort(key=lambda x: x[2], reverse=bool(descending))

    if max_rows is not None:
        max_rows_i = int(max_rows)
        if max_rows_i <= 0:
            raise ValueError(f"format_timing_report: max_rows must be > 0, got {max_rows_i}.")
        shown = rows[:max_rows_i]
        hidden = int(max(0, len(rows) - len(shown)))
    else:
        shown = rows
        hidden = 0

    name_w = max(4, *(len(r[0]) for r in shown)) if len(shown) > 0 else 4
    sec_w = 11
    pct_w = 6
    cnt_w = 5
    header = (
        f"{'span':<{name_w}} "
        f"{'seconds':>{sec_w}} "
        f"{'share':>{pct_w}} "
        f"{'count':>{cnt_w}}"
    )
    total = float(timing.total_seconds)

    lines: list[str] = [str(title), header, "-" * len(header)]
    for name, seconds, count in shown:
        pct = 100.0 * seconds / total if total > 0.0 else 0.0
        lines.append(
            f"{name:<{name_w}} "
            f"{seconds:>{sec_w}.3e} "
            f"{pct:>{pct_w}.1f}% "
            f"{count:>{cnt_w}d}"
        )
    if hidden > 0:
        lines.append(f"... (+{hidden} more spans)")
    lines.append(
        f"{'total':<{name_w}} "
        f"{total:>{sec_w}.3e} "
        f"{100.0:>{pct_w}.1f}% "
        f"{'-':>{cnt_w}}"
    )
    return "\n".join(lines)


def _append_timing_summary(lines: list[str], timing: TimingReport) -> None:
    lines.extend(
        format_timing_report(
            timing,
            title="Timing",
            sort_by="seconds",
            descending=True,
        ).splitlines()
    )
    lines.append("")


def format_solve_report(
    runtime: NLSRuntime,
    *,
    x0: Array | None = None,
    x_star: Array | None = None,
    outcome: SolveOutcome | None = None,
    required: Iterable[StateKey] | None = None,
    max_elems: int = 6,
    precision: int = 4,
    include_vars: bool = True,
    include_named: bool = True,
    include_diagnostics: bool = True,
    include_kkt: bool = False,
    kkt_kwargs: Mapping[str, Any] | None = None,
    solve_summary: Mapping[str, Any] | None = None,
    trajectory_summary: Mapping[str, Any] | None = None,
    active_tol: float = 1e-10,
    name_blacklist: set[str] | None = None,
) -> str:
    """Format a concise post-solve report of objective terms and expression values.

    The report has three sections:
      - Solve/trajectory summary (optional)
      - Variables (x0 / x* if provided)
      - Term residuals and per-term cost contributions (after cost weighting)
      - Diagnostics (`||J^T r||`, `rank(J)`, singular values, active terms)
      - KKT residual checks (optional)
      - Named expression values (skipping default/generic names by default)
    """

    if name_blacklist is None:
        name_blacklist = set(_DEFAULT_NAME_BLACKLIST)

    problem = runtime.problem
    pack = runtime.pack
    required_list = runtime.required_list(required)
    runtime.update_state_if_needed(required=required_list)

    lines: list[str] = []

    if outcome is not None and x_star is None:
        x_star = np.asarray(outcome.solution, dtype=float).reshape(-1).copy()
    if outcome is not None and solve_summary is None:
        solve_summary = outcome.to_summary()
    if x0 is None and outcome is not None:
        x0_meta = outcome.meta.get("x0", None)
        if x0_meta is not None:
            x0 = np.asarray(x0_meta, dtype=float).reshape(-1).copy()

    if solve_summary is not None:
        _append_solve_summary(lines, solve_summary)
    if outcome is not None:
        _append_timing_summary(lines, outcome.timing)

    if trajectory_summary is not None:
        _append_trajectory_summary(lines, trajectory_summary)

    if include_vars:
        x_star_vec = np.asarray(x_star if x_star is not None else pack.get(), dtype=float).reshape(-1)
        expected = int(getattr(pack, "n_total", x_star_vec.size))
        if x_star_vec.size != expected:
            raise ValueError(f"format_solve_report: x_star has size {x_star_vec.size}, expected {expected}.")

        x0_vec = None if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        if x0_vec is not None and x0_vec.size != x_star_vec.size:
            raise ValueError(f"format_solve_report: x0 has size {x0_vec.size}, expected {x_star_vec.size}.")

        lines.append("Variables:")

        slices = getattr(pack, "slices", {})
        for v in getattr(pack, "vars", []) or []:
            name = getattr(v, "name", None)
            if not isinstance(name, str) or not name:
                continue
            if name not in slices:
                continue

            s, e = slices[name]
            xs_v = x_star_vec[int(s) : int(e)]

            if x0_vec is None:
                lines.append(f"- {name}: x*={_format_vector(xs_v, max_elems=max_elems, precision=precision)}")
                continue

            x0_v = x0_vec[int(s) : int(e)]
            lines.append(
                f"- {name}: "
                f"x0={_format_vector(x0_v, max_elems=max_elems, precision=precision)} | "
                f"x*={_format_vector(xs_v, max_elems=max_elems, precision=precision)}"
            )

        lines.append("")

    # Term summary (objective contributions)
    term_linear_raw = runtime.linearize_terms(required=required_list, weighted=False)
    term_linear_weighted = runtime.linearize_terms(required=required_list, weighted=True)
    if len(term_linear_raw) != len(term_linear_weighted):
        raise RuntimeError(
            "format_solve_report: internal term linearization mismatch between raw and weighted terms."
        )

    lines.append("Terms:")
    total_cost = 0.0
    active_terms: list[tuple[int, str, float]] = []
    for term_raw, term_w in zip(term_linear_raw, term_linear_weighted):
        i = int(term_raw.term_index)
        expr, cost = problem.terms[i]
        r = np.asarray(term_raw.residual, dtype=float).reshape(-1)
        r_w = np.asarray(term_w.residual, dtype=float).reshape(-1)
        cost_i = float(r_w @ r_w)
        total_cost += cost_i

        cost_name = getattr(cost, "name", cost.__class__.__name__)
        rnorm = float(np.linalg.norm(r))
        rnorm_w = float(np.linalg.norm(r_w))
        if rnorm_w > float(active_tol):
            active_terms.append((i, str(getattr(expr, "name", expr.__class__.__name__)), rnorm_w))

        r_str = _format_vector(r, max_elems=max_elems, precision=precision)
        lines.append(
            f"- [{i}] {getattr(expr, 'name', expr.__class__.__name__)} | cost={cost_name} | "
            f"||r||={rnorm:.3e} | ||r_w||={rnorm_w:.3e} | cost_i={cost_i:.3e} | r={r_str}"
        )

    lines.append(f"Total cost: {total_cost:.3e}")

    if include_diagnostics:
        if len(problem.terms) == 0:
            J_all = np.zeros((0, int(runtime.pack.n_total)), dtype=float)
            grad_norm = 0.0
            rank = 0
            svals = np.zeros((0,), dtype=float)
            svals_str = _format_vector(svals, max_elems=max_elems, precision=precision)
        else:
            r_all, J_all = runtime.linearize(required=required_list)
            grad = np.asarray(J_all.T @ r_all, dtype=float).reshape(-1)
            grad_norm = float(np.linalg.norm(grad))
            rank = int(np.linalg.matrix_rank(J_all)) if J_all.size > 0 else 0
            svals = np.linalg.svd(J_all, compute_uv=False) if J_all.size > 0 else np.zeros((0,), dtype=float)
            svals_str = _format_vector(svals, max_elems=max_elems, precision=precision)

        if len(active_terms) == 0:
            active_str = "none"
        else:
            active_str = ", ".join(f"[{i}] {name} ({norm:.3e})" for i, name, norm in active_terms)

        lines.append("")
        lines.append("Diagnostics:")
        lines.append(f"- ||J^T r||={grad_norm:.3e}")
        lines.append(f"- rank(J)={rank}/{int(J_all.shape[1])} (rows={int(J_all.shape[0])})")
        lines.append(f"- svd(J)={svals_str}")
        lines.append(f"- active terms (||r_w||>{float(active_tol):.1e}): {active_str}")

    if include_kkt:
        kwargs_local: dict[str, Any] = {} if kkt_kwargs is None else dict(kkt_kwargs)
        if "required" in kwargs_local:
            raise ValueError("format_solve_report: pass `required` via function argument, not via kkt_kwargs.")
        kkt_out = check_kkt_conditions(
            runtime,
            required=required_list,
            **kwargs_local,
        )
        _append_kkt_report(lines, kkt_out)

    if include_named:
        lines.append("")
        lines.append("Named expr values:")
        named_values = collect_named_expr_values(
            runtime,
            required=required_list,
            name_blacklist=name_blacklist,
            include_blacklisted=False,
        )
        for item in named_values:
            y_str = _format_vector(item.value, max_elems=max_elems, precision=precision)
            lines.append(f"- [t{item.term_index}] {item.name} ({item.expr_type}): {y_str}")

    return "\n".join(lines)
