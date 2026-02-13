"""Backend integrations.

The core of `eiopt` is backend-agnostic: you connect a robotics/physics backend via a
`build_state()` function (see `eiopt.core.StateCache`).

This package contains optional helpers for specific ecosystems (e.g. Pinocchio).
They should not be imported unless you have the corresponding dependency installed.
"""

from __future__ import annotations

from .trajectory_adapter import (
    BackendTrajectoryCompileResult,
    TrajectoryBackendAdapter,
    compile_trajectory_problem_with_adapter,
)

__all__ = [
    "BackendTrajectoryCompileResult",
    "TrajectoryBackendAdapter",
    "compile_trajectory_problem_with_adapter",
]
