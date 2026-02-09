from __future__ import annotations

from collections.abc import Iterator
from typing import Any


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


def find_var_spec(dsl: dict, *, name: str) -> dict | None:
    """Find a variable entry in `dsl["variables"]` by its `name` field."""

    for v in dsl.get("variables", []) or []:
        if isinstance(v, dict) and v.get("name", None) == name:
            return v
    return None

