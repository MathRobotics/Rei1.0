"""State-backend namespace.

This package intentionally does not import optional backend modules at import
time. Import concrete modules directly, e.g.:

- ``eiopt.backends.state.dispatch.template``
- ``eiopt.backends.state.dispatch.composite``
- ``eiopt.backends.state.robotics.kots``
- ``eiopt.backends.state.robotics.pinocchio``
- ``eiopt.backends.state.vision.provider``
- ``eiopt.backends.state.vision.pinhole``
"""

from __future__ import annotations

__all__ = [
    "dispatch",
    "robotics",
    "vision",
]
