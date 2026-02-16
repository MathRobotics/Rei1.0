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
    default_var: str | None = None
    vars_dsl = data.get("variables")
    if isinstance(vars_dsl, list):
        names = [v.get("name") for v in vars_dsl if isinstance(v, dict) and "name" in v]
        names = [str(n) for n in names if n is not None]
        if len(names) == 1:
            default_var = names[0]

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
            if isinstance(jac, dict):
                if "var" not in jac and default_var is not None:
                    jac["var"] = default_var
                continue
            if default_var is not None:
                node["jac"] = {"var": default_var}


__all__ = [
    "load_problem_toml",
]
