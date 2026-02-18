"""State-backend namespace.

This package intentionally does not import optional backend modules at import
time. Import concrete modules directly, e.g.:

- ``eiopt.backends.state.kots``
- ``eiopt.backends.state.pinocchio``
- ``eiopt.backends.state.composite``
- ``eiopt.backends.state.vision``
- ``eiopt.backends.state.vision_pinhole``
"""

from __future__ import annotations

__all__ = [
    "template",
    "spatial",
    "composite",
    "vision",
    "vision_pinhole",
    "kots",
    "pinocchio",
]
