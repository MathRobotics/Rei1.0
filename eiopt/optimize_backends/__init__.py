"""Optimize-backend integration namespace.

This package separates backend state implementations from optimize-specific
compile helpers.
"""

from __future__ import annotations

from .problem_adapter import (
    BackendCompileResult,
    ProblemBackendAdapter,
    compile_problem_with_adapter,
)
from .trajectory_adapter import (
    BackendTrajectoryCompileResult,
    TrajectoryBackendAdapter,
    compile_trajectory_problem_with_adapter,
)

__all__ = [
    "BackendCompileResult",
    "ProblemBackendAdapter",
    "compile_problem_with_adapter",
    "BackendTrajectoryCompileResult",
    "TrajectoryBackendAdapter",
    "compile_trajectory_problem_with_adapter",
    "problem_adapter",
    "vision",
    "kots",
    "pinocchio",
]
