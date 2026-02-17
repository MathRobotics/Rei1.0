from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np


def iter_nodes(obj: Any) -> Iterator[dict]:
    """Yield all dict nodes inside a nested (dict/list) structure."""

    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from iter_nodes(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_nodes(item)


def rewrite_get_state_owner_name(
    dsl: dict,
    *,
    dtype: str,
    owner_type: str,
    owner_name: str,
) -> int:
    """Rewrite `key.owner_name` for matching `get_state` expressions.

    Matches:
      - node.type == "get_state"
      - node.key.dtype == dtype
      - node.key.owner_type == owner_type
    """

    n = 0
    for term in dsl.get("terms", []) or []:
        expr = term.get("expr", None)
        for node in iter_nodes(expr):
            if node.get("type", None) != "get_state":
                continue
            key = node.get("key", None)
            if not isinstance(key, dict):
                continue
            if key.get("dtype", None) != dtype:
                continue
            if key.get("owner_type", None) != owner_type:
                continue
            key["owner_name"] = str(owner_name)
            n += 1
    return n


def find_const_expr(dsl: dict, *, name: str) -> dict | None:
    """Find a `const` expression node by its `name` field."""

    for term in dsl.get("terms", []) or []:
        expr = term.get("expr", None)
        for node in iter_nodes(expr):
            if node.get("type", None) == "const" and node.get("name", None) == name:
                return node
    return None


def find_var_dsl(dsl: dict, *, name: str) -> dict | None:
    """Find a variable entry in `dsl["variables"]` by its `name` field."""

    for v in dsl.get("variables", []) or []:
        if isinstance(v, dict) and v.get("name", None) == name:
            return v
    return None


def _normalize_indices(
    *,
    size: int,
    indices: Iterable[int] | None,
    where: str,
) -> tuple[int, ...]:
    n = int(size)
    if n <= 0:
        raise ValueError(f"{where}: size must be > 0, got {n}.")
    if indices is None:
        return tuple(range(n))

    out: list[int] = []
    seen: set[int] = set()
    for raw in indices:
        idx = int(raw)
        if idx < 0 or idx >= n:
            raise ValueError(f"{where}: index out of range: {idx}. Expected 0..{n - 1}.")
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    if len(out) == 0:
        raise ValueError(f"{where}: indices must be non-empty.")
    return tuple(out)


def _split_cost_dsl_for_component(
    cost_dsl: dict[str, Any],
    *,
    segment_dim: int,
    index: int,
) -> dict[str, Any]:
    out = copy.deepcopy(cost_dsl)
    ctype = str(out.get("type", "")).strip().lower()
    if ctype != "diag_weight":
        return out
    if "w" not in out:
        return out

    w = np.asarray(out["w"], dtype=float).reshape(-1)
    if w.size <= 1:
        return out
    seg = int(segment_dim)
    if w.size % seg != 0:
        raise ValueError(
            "split_terms_by_component: diag_weight size mismatch. "
            f"w.size={w.size} is not divisible by segment_dim={seg}."
        )

    out["w"] = w.reshape(-1, seg)[:, int(index)].reshape(-1).tolist()
    return out


def split_terms_by_component(
    dsl: dict,
    *,
    segment_dim: int,
    component_indices: list[int] | tuple[int, ...] | None = None,
    term_indices: list[int] | tuple[int, ...] | None = None,
) -> int:
    """Expand selected terms into per-component terms using ``expr.type='component'``.

    The input DSL is mutated in-place by replacing selected terms with
    ``len(component_indices)`` duplicated terms, each wrapped by a component selector.
    Returns the number of original terms that were expanded.
    """

    seg = int(segment_dim)
    if seg <= 0:
        raise ValueError(f"split_terms_by_component: segment_dim must be > 0, got {seg}.")

    terms = dsl.get("terms", None)
    if not isinstance(terms, list):
        raise ValueError("split_terms_by_component: dsl['terms'] must be a list.")

    comp_idxs = _normalize_indices(
        size=seg,
        indices=component_indices,
        where="split_terms_by_component.component_indices",
    )
    tgt_idxs = _normalize_indices(
        size=len(terms),
        indices=term_indices,
        where="split_terms_by_component.term_indices",
    )
    tgt_set = set(tgt_idxs)

    expanded = 0
    out_terms: list[dict[str, Any]] = []
    for i, term in enumerate(terms):
        if i not in tgt_set:
            out_terms.append(term)
            continue
        if not isinstance(term, dict):
            raise ValueError(
                f"split_terms_by_component: terms[{i}] must be a dict, got {type(term).__name__}."
            )
        expr = term.get("expr", None)
        if not isinstance(expr, dict):
            raise ValueError(f"split_terms_by_component: terms[{i}].expr must be a dict.")

        base_name = str(expr.get("name", f"term_{i}"))
        for j in comp_idxs:
            term_j = copy.deepcopy(term)
            term_j["expr"] = {
                "type": "component",
                "name": f"{base_name}_j{j}",
                "segment_dim": seg,
                "index": int(j),
                "base": copy.deepcopy(expr),
            }
            cost_dsl = term_j.get("cost", None)
            if isinstance(cost_dsl, dict):
                term_j["cost"] = _split_cost_dsl_for_component(
                    cost_dsl,
                    segment_dim=seg,
                    index=int(j),
                )

            attrs = term_j.get("attrs", None)
            if attrs is None:
                attrs = {}
            elif not isinstance(attrs, dict):
                raise ValueError(f"split_terms_by_component: terms[{i}].attrs must be a dict when provided.")
            attrs = dict(attrs)
            attrs["joint_component"] = int(j)
            term_j["attrs"] = attrs
            out_terms.append(term_j)
        expanded += 1

    dsl["terms"] = out_terms
    return expanded


__all__ = [
    "iter_nodes",
    "rewrite_get_state_owner_name",
    "find_const_expr",
    "find_var_dsl",
    "split_terms_by_component",
]
