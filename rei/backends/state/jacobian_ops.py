from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

Array = np.ndarray


@runtime_checkable
class DenseJacobianProvider(Protocol):
    def jacobian(self, state_ref: Any) -> Array:
        """Return a dense Jacobian for `state_ref`."""


@runtime_checkable
class JvpProvider(Protocol):
    def jacobian_mul(self, state_ref: Any, cols: Array) -> Array:
        """Return `J(state_ref) @ cols` for vector or matrix `cols`."""


@runtime_checkable
class VjpProvider(Protocol):
    def jacobian_transpose_mul(self, state_ref: Any, rhs: Array) -> Array:
        """Return `J(state_ref).T @ rhs` for vector or matrix `rhs`."""


class JacobianOperator(Protocol):
    def dense(self, state_ref: Any) -> Array:
        """Return a dense Jacobian."""

    def jvp(self, state_ref: Any, cols: Array, *, value_size: int | None = None) -> Array:
        """Return `J @ cols` without requiring callers to know backend fallback details."""

    def vjp(self, state_ref: Any, rhs: Array) -> Array:
        """Return `J.T @ rhs`."""


def as_2d_numeric_matrix(value: Any, *, name: str) -> Array:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}.")
    return arr


def normalize_jvp_output(out_raw: Any, *, input_cols: Array, backend_name: str) -> Array:
    out = np.asarray(out_raw, dtype=float)
    C = as_2d_numeric_matrix(input_cols, name=f"{backend_name} JVP input columns")
    if out.ndim == 1 and C.shape[1] == 1:
        return out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"{backend_name} JVP output must be 2D, got shape {out.shape}.")
    if out.shape[1] == C.shape[1]:
        return out
    if out.shape[0] == C.shape[1]:
        return out.T
    raise ValueError(
        f"{backend_name} JVP output column mismatch. "
        f"Expected {C.shape[1]} columns, got shape {out.shape}."
    )


__all__ = [
    "DenseJacobianProvider",
    "JacobianOperator",
    "JvpProvider",
    "VjpProvider",
    "as_2d_numeric_matrix",
    "normalize_jvp_output",
]
