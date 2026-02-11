from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core.state_cache import StateKey
from .model.runtime import ProblemRuntime

Array = np.ndarray

_DEFAULT_NAME_BLACKLIST = {
    "const",
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
    runtime: ProblemRuntime,
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
    runtime: ProblemRuntime,
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


def format_solve_report(
    runtime: ProblemRuntime,
    *,
    x0: Array | None = None,
    x_star: Array | None = None,
    required: Iterable[StateKey] | None = None,
    max_elems: int = 6,
    precision: int = 4,
    include_vars: bool = True,
    include_named: bool = True,
    name_blacklist: set[str] | None = None,
) -> str:
    """Format a concise post-solve report of objective terms and expression values.

    The report has three sections:
      - Variables (x0 / x* if provided)
      - Term residuals and per-term cost contributions (after cost weighting)
      - Named expression values (skipping default/generic names by default)
    """

    if name_blacklist is None:
        name_blacklist = set(_DEFAULT_NAME_BLACKLIST)

    problem = runtime.problem
    ctx = runtime.ctx
    pack = runtime.pack
    required_list = runtime.required_list(required)
    runtime.update_state_if_needed(required=required_list)

    lines: list[str] = []

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
    lines.append("Terms:")
    total_cost = 0.0
    for i, (expr, cost) in enumerate(problem.terms):
        r, blocks = expr.eval(ctx)
        r = np.asarray(r, dtype=float).reshape(-1)
        blocks = [np.asarray(B, dtype=float) for B in blocks]

        apply_cost = getattr(cost, "apply", None)
        if not callable(apply_cost):
            raise TypeError(f"Cost object for term {i} has no callable apply(r, blocks).")

        r_w, _blocks_w = apply_cost(r, blocks)
        r_w = np.asarray(r_w, dtype=float).reshape(-1)

        cost_i = float(r_w @ r_w)
        total_cost += cost_i

        cost_name = getattr(cost, "name", cost.__class__.__name__)
        rnorm = float(np.linalg.norm(r))
        rnorm_w = float(np.linalg.norm(r_w))

        r_str = _format_vector(r, max_elems=max_elems, precision=precision)
        lines.append(
            f"- [{i}] {getattr(expr, 'name', expr.__class__.__name__)} | cost={cost_name} | "
            f"||r||={rnorm:.3e} | ||r_w||={rnorm_w:.3e} | cost_i={cost_i:.3e} | r={r_str}"
        )

    lines.append(f"Total cost: {total_cost:.3e}")

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
