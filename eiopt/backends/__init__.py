"""Backend integrations.

The core of `eiopt` is backend-agnostic: you connect a robotics/physics backend via a
`build_state()` function (see `eiopt.core.StateCache`).

This package contains optional helpers for specific ecosystems (e.g. Pinocchio).
They should not be imported unless you have the corresponding dependency installed.
"""

from __future__ import annotations

__all__ = []

