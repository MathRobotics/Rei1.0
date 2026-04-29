from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..core.state_cache import StateKey
from .caps import Array


def runtime_pack(runtime: Any, *, adapter_name: str) -> Any:
    pack = getattr(runtime, "pack", None)
    if pack is None:
        raise AttributeError(f"{adapter_name}: runtime must expose `.pack`.")
    return pack


def runtime_n_total(runtime: Any, *, adapter_name: str) -> int:
    pack = runtime_pack(runtime, adapter_name=adapter_name)
    n_total = getattr(pack, "n_total", None)
    if n_total is None:
        raise AttributeError(f"{adapter_name}: runtime.pack must expose `.n_total`.")
    return int(n_total)


def runtime_point(runtime: Any, *, adapter_name: str) -> Array:
    pack = runtime_pack(runtime, adapter_name=adapter_name)
    get = getattr(pack, "get", None)
    if not callable(get):
        raise AttributeError(f"{adapter_name}: runtime.pack must expose callable get().")
    return np.asarray(get(), dtype=float).reshape(-1).copy()


def runtime_required_list(
    runtime: Any,
    required: Iterable[StateKey] | None,
) -> list[StateKey]:
    fn = getattr(runtime, "required_list", None)
    if callable(fn):
        return list(fn(required))
    if required is None:
        return []
    return list(required)


__all__ = [
    "runtime_pack",
    "runtime_n_total",
    "runtime_point",
    "runtime_required_list",
]
