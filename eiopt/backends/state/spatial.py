from __future__ import annotations

from typing import Literal

import numpy as np

Array = np.ndarray

Jacobian6Order = Literal["linear_angular", "angular_linear"]


def as_6xn_jacobian(J6: Array) -> Array:
    """Normalize a 6D Jacobian to shape (6, n)."""

    J = np.asarray(J6, dtype=float)
    if J.ndim != 2:
        raise ValueError(f"Expected 2D Jacobian, got {J.shape}.")
    if J.shape[0] == 6:
        return J
    if J.shape[1] == 6:
        return J.T
    raise ValueError(f"Expected Jacobian shape (6,n) or (n,6), got {J.shape}.")


def split_jacobian6(J6: Array, *, order: Jacobian6Order) -> tuple[Array, Array]:
    """Split 6D Jacobian into (linear, angular) blocks (each is (3, n))."""

    J = as_6xn_jacobian(J6)
    if order == "linear_angular":
        return J[:3, :], J[3:, :]
    if order == "angular_linear":
        return J[3:, :], J[:3, :]
    raise ValueError(f"Unknown Jacobian6Order: {order!r} (expected 'linear_angular' or 'angular_linear').")


def linear_part_from_jacobian6(J6: Array, *, order: Jacobian6Order) -> Array:
    J_lin, _J_ang = split_jacobian6(J6, order=order)
    return J_lin


def angular_part_from_jacobian6(J6: Array, *, order: Jacobian6Order) -> Array:
    _J_lin, J_ang = split_jacobian6(J6, order=order)
    return J_ang


__all__ = [
    "Jacobian6Order",
    "as_6xn_jacobian",
    "split_jacobian6",
    "linear_part_from_jacobian6",
    "angular_part_from_jacobian6",
]
