"""Vision-oriented state providers and helpers.

This package also preserves the historical import style:

    from rei.backends.state.vision import CameraCalibrationStateProvider
"""

from __future__ import annotations

from .provider import CameraCalibrationStateProvider, ModelUpdateFn, VisionFieldHandler
from .pinhole import (
    PINHOLE_RADIAL_PARAM_ORDER,
    PinholeRadialParameterOrder,
    build_pinhole_radial_vision_field_handler,
    pinhole_radial_reprojection,
    pinhole_radial_reprojection_jacobian,
)

__all__ = [
    "provider",
    "pinhole",
    "VisionFieldHandler",
    "CameraCalibrationStateProvider",
    "ModelUpdateFn",
    "PinholeRadialParameterOrder",
    "PINHOLE_RADIAL_PARAM_ORDER",
    "pinhole_radial_reprojection",
    "pinhole_radial_reprojection_jacobian",
    "build_pinhole_radial_vision_field_handler",
]
