from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

Array = np.ndarray


def resolve_variable_dim(raw: Any, *, name: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        return None
    try:
        dim = int(raw)
    except Exception as e:
        raise ValueError(f"variable '{name}': dim must be an integer, got {raw!r}.") from e
    if dim <= 0:
        raise ValueError(f"variable '{name}': dim must be > 0, got {dim}.")
    return dim


def expand_variable_init(raw: Any, *, dim: int | None, where: str) -> Array:
    if isinstance(raw, Mapping):
        if "fill" not in raw:
            raise ValueError(f"{where}: dict init must contain 'fill'.")
        if dim is None:
            raise ValueError(f"{where}: init.fill requires explicit dim.")
        try:
            fill_value = float(raw["fill"])
        except Exception as e:
            raise ValueError(f"{where}: init.fill must be numeric, got {raw['fill']!r}.") from e
        return np.full((int(dim),), fill_value, dtype=float)

    x_vec = np.asarray(raw, dtype=float).reshape(-1)
    if dim is None:
        return x_vec
    dim_i = int(dim)
    if dim_i <= 0:
        raise ValueError(f"{where}: dim must be > 0, got {dim_i}.")
    if x_vec.size == dim_i:
        return x_vec.copy()
    if x_vec.size == 1 and dim_i > 1:
        raise ValueError(
            f"{where}: scalar init is not broadcast implicitly. "
            "Use `init = { fill = <value> }` to fill all elements."
        )
    raise ValueError(f"{where}: init size {x_vec.size} is not compatible with dim {dim_i}.")


def normalize_variable_dsl(
    var_dsl: Mapping[str, Any],
    *,
    expected_dim: int,
    where: str,
) -> dict[str, Any]:
    out = dict(var_dsl)
    name = str(out.get("name", "")).strip()
    if name == "":
        raise ValueError(f"{where}: variable name must be non-empty.")

    dim_expected = int(expected_dim)
    if dim_expected <= 0:
        raise ValueError(f"{where}: expected_dim must be > 0, got {dim_expected}.")

    dim = resolve_variable_dim(out.get("dim", None), name=name)
    if dim is None:
        dim = dim_expected
    elif int(dim) != dim_expected:
        raise ValueError(f"{where} dim mismatch: dsl={int(dim)}, expected={dim_expected}.")

    init_raw = out.get("init", None)
    if init_raw is None:
        init_vec = np.zeros((dim_expected,), dtype=float)
    else:
        init_vec = expand_variable_init(init_raw, dim=dim_expected, where=f"{where}.init")

    out["name"] = name
    out["dim"] = int(dim_expected)
    out["init"] = np.asarray(init_vec, dtype=float).reshape(-1).tolist()
    return out

