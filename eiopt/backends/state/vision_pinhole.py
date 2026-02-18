from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .vision import VisionFieldHandler

Array = np.ndarray


@dataclass(frozen=True)
class PinholeRadialParameterOrder:
    """Parameter order for pinhole + radial distortion model."""

    fx: int = 0
    fy: int = 1
    cx: int = 2
    cy: int = 3
    k1: int = 4
    k2: int = 5

    @property
    def size(self) -> int:
        return 6

    @property
    def names(self) -> tuple[str, ...]:
        return ("fx", "fy", "cx", "cy", "k1", "k2")


PINHOLE_RADIAL_PARAM_ORDER = PinholeRadialParameterOrder()


def _as_points_xy(points_xy: Array | Any) -> Array:
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            "pinhole radial model: points_xy must have shape (N,2). "
            f"Got {pts.shape}."
        )
    if pts.shape[0] <= 0:
        raise ValueError("pinhole radial model: points_xy must be non-empty.")
    return pts


def _as_theta(theta: Array | Any) -> Array:
    q = np.asarray(theta, dtype=float).reshape(-1)
    if q.size != PINHOLE_RADIAL_PARAM_ORDER.size:
        raise ValueError(
            "pinhole radial model: theta size mismatch. "
            f"Expected {PINHOLE_RADIAL_PARAM_ORDER.size}, got {q.size}."
        )
    return q


def pinhole_radial_reprojection(
    theta: Array | Any,
    points_xy: Array | Any,
) -> Array:
    """Project normalized points with pinhole + radial(k1,k2) model.

    theta = [fx, fy, cx, cy, k1, k2]
    points_xy: normalized coordinates (x, y) with shape (N, 2)
    returns: flattened [u0, v0, u1, v1, ...] with shape (2N,)
    """

    q = _as_theta(theta)
    pts = _as_points_xy(points_xy)

    x = pts[:, 0]
    y = pts[:, 1]
    r2 = x * x + y * y
    g = 1.0 + q[4] * r2 + q[5] * r2 * r2

    u = q[0] * x * g + q[2]
    v = q[1] * y * g + q[3]

    out = np.empty((2 * pts.shape[0],), dtype=float)
    out[0::2] = u
    out[1::2] = v
    return out


def pinhole_radial_reprojection_jacobian(
    theta: Array | Any,
    points_xy: Array | Any,
) -> Array:
    """Analytic Jacobian of `pinhole_radial_reprojection` w.r.t theta."""

    q = _as_theta(theta)
    pts = _as_points_xy(points_xy)
    n = int(pts.shape[0])
    J = np.zeros((2 * n, PINHOLE_RADIAL_PARAM_ORDER.size), dtype=float)

    x = pts[:, 0]
    y = pts[:, 1]
    r2 = x * x + y * y
    r4 = r2 * r2
    g = 1.0 + q[4] * r2 + q[5] * r4

    # u = fx * x * g + cx
    J[0::2, 0] = x * g  # du/dfx
    J[0::2, 2] = 1.0  # du/dcx
    J[0::2, 4] = q[0] * x * r2  # du/dk1
    J[0::2, 5] = q[0] * x * r4  # du/dk2

    # v = fy * y * g + cy
    J[1::2, 1] = y * g  # dv/dfy
    J[1::2, 3] = 1.0  # dv/dcy
    J[1::2, 4] = q[1] * y * r2  # dv/dk1
    J[1::2, 5] = q[1] * y * r4  # dv/dk2
    return J


def build_pinhole_radial_vision_field_handler(
    *,
    points_xy: Array | Any,
) -> VisionFieldHandler:
    pts = _as_points_xy(points_xy)

    def _value(q: Array, key: Any, state_ref: Any) -> Array:
        del key, state_ref
        return pinhole_radial_reprojection(q, pts)

    def _jac(q: Array, key: Any, state_ref: Any) -> Array:
        del key, state_ref
        return pinhole_radial_reprojection_jacobian(q, pts)

    return VisionFieldHandler(
        value_handler=_value,
        jac_handler=_jac,
    )


__all__ = [
    "PinholeRadialParameterOrder",
    "PINHOLE_RADIAL_PARAM_ORDER",
    "pinhole_radial_reprojection",
    "pinhole_radial_reprojection_jacobian",
    "build_pinhole_radial_vision_field_handler",
]
