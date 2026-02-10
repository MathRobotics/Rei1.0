from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from .core.state_cache import StateKey
from .model.runtime import ProblemRuntime

Array = np.ndarray

_DEFAULT_NAME_BLACKLIST = {
    "const",
    "get_state",
    "get_var",
    "hinge",
    "stack",
    "sub",
}

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

        for i, (expr, _cost) in enumerate(problem.terms):
            for node in _walk_expr(expr):
                name = getattr(node, "name", None)
                if not isinstance(name, str) or not name:
                    continue
                if name in name_blacklist:
                    continue

                y, _blocks = node.eval(ctx)
                y = np.asarray(y, dtype=float).reshape(-1)
                y_str = _format_vector(y, max_elems=max_elems, precision=precision)
                lines.append(f"- [t{i}] {name} ({node.__class__.__name__}): {y_str}")

    return "\n".join(lines)
