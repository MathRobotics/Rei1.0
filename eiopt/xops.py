from __future__ import annotations

from typing import Any

import numpy as np

from .core.expr.types import VariablePack

Array = np.ndarray


def as_vec(x: Array | Any, *, expected_size: int, name: str) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size != int(expected_size):
        raise ValueError(f"{name}: expected size={int(expected_size)}, got size={v.size}.")
    return v


def set_pack_x(pack: VariablePack, x: Array | Any, *, name: str = "x") -> None:
    x_new = as_vec(x, expected_size=int(pack.n_total), name=name)
    x_cur = np.asarray(pack.get(), dtype=float).reshape(-1)
    if np.array_equal(x_cur, x_new):
        return
    pack.apply_dx(x_new - x_cur)


def set_runtime_x(runtime: Any, x: Array | Any, *, name: str = "x") -> None:
    set_pack_x(runtime.pack, x, name=name)


def apply_pack_dx(pack: VariablePack, dx: Array | Any, *, name: str = "dx") -> None:
    dx_vec = as_vec(dx, expected_size=int(pack.n_total), name=name)
    pack.apply_dx(dx_vec)


__all__ = [
    "as_vec",
    "set_pack_x",
    "set_runtime_x",
    "apply_pack_dx",
]
