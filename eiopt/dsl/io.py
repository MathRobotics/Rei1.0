from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib

from .dsl_ops import iter_nodes


def load_problem_toml(path: str | Path) -> dict[str, Any]:
    """Load a problem definition from a TOML file."""

    p = Path(path)
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Problem TOML must decode to a dict.")
    _normalize_toml_problem_def(data)
    return data


def _normalize_toml_problem_def(data: dict[str, Any]) -> None:
    terms = data.get("terms")
    if not isinstance(terms, list):
        return
    for term in terms:
        if not isinstance(term, dict):
            continue
        expr = term.get("expr")
        if not isinstance(expr, dict):
            continue
        for node in iter_nodes(expr):
            if node.get("type") != "get_state":
                continue
            jac = node.get("jac")
            if isinstance(jac, dict) and "var" in jac:
                continue
            node["jac"] = {"var": "q"}
