"""State-backend namespace.

This package intentionally does not import optional backend modules at import
time. Import concrete modules directly, e.g.:

- ``eiopt.backends.state.kots``
- ``eiopt.backends.state.pinocchio``
"""

from __future__ import annotations

__all__ = [
    "template",
    "spatial",
    "kots",
    "pinocchio",
]
