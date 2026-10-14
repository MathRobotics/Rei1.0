"""Traverse expression graphs through their optional children() capability."""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import fields, is_dataclass
from typing import Any


def _nested_expressions(value: Any) -> Iterator[Any]:
    if callable(getattr(value, "eval", None)):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _nested_expressions(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _nested_expressions(child)


def iter_expr_children(expr: Any) -> Iterable[Any]:
    """Return direct children, retaining compatibility with existing custom nodes.

    New composite expressions should implement children(). Older dataclass or
    conventional a/b/base/inner/parts nodes remain traversable without changes.
    """
    children = getattr(expr, "children", None)
    if callable(children):
        yield from children()
        return
    names = (
        (item.name for item in fields(expr)) if is_dataclass(expr)
        else ("a", "b", "base", "inner", "parts", "torque", "velocity")
    )
    for name in names:
        yield from _nested_expressions(getattr(expr, name, None))


def walk_expr(expr: Any) -> Iterator[Any]:
    """Visit nodes in depth-first order, once per identity, including the root."""
    stack = [expr]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(reversed(list(iter_expr_children(node))))
