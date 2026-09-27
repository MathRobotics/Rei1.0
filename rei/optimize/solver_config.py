"""Solver defaults carried by compiled runtimes, independently of residuals."""
from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def normalize_solver_config(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError('solver must be a table with name and optional options.')
    unknown = set(value) - {'name', 'options'}
    if unknown:
        raise ValueError(f'Unknown solver field(s): {sorted(unknown)}; put settings in solver.options.')
    name = value.get('name')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('solver.name must be a non-empty string.')
    options = value.get('options', {})
    if not isinstance(options, Mapping):
        raise ValueError('solver.options must be a table.')
    return {'name': name.strip().lower(), 'options': deepcopy(dict(options))}
