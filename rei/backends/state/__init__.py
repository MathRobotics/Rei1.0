"""State-backend namespace.

This package intentionally does not import optional backend modules at import
time. Import concrete modules directly, e.g.:

- ``rei.backends.state.dispatch.template``
- ``rei.backends.state.dispatch.composite``
- ``rei.backends.state.robotics.kots``
- ``rei.backends.state.robotics.pinocchio``
- ``rei.backends.state.vision.provider``
- ``rei.backends.state.vision.pinhole``
"""

from __future__ import annotations

__all__ = [
    "dispatch",
    "robotics",
    "vision",
]
