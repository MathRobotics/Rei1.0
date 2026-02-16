"""Optimize-backend integration namespace.

This package separates backend state implementations from optimize-specific
compile helpers.
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
    "kots",
    "pinocchio",
]
