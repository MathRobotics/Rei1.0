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
from .trajectory_diagnostics import (
    BackendFieldCapability,
    TrajectoryProblemDiagnostics,
    UnsupportedTermDiagnostic,
    inspect_trajectory_problem_backend,
)
from .trajectory_ioc import (
    TrajectoryIocCompiledProblem,
    compile_trajectory_ioc_problem,
    estimate_ioc_weights,
)

__all__ = [
    "BackendCompileResult",
    "ProblemBackendAdapter",
    "compile_problem_with_adapter",
    "BackendTrajectoryCompileResult",
    "TrajectoryBackendAdapter",
    "compile_trajectory_problem_with_adapter",
    "BackendFieldCapability",
    "TrajectoryProblemDiagnostics",
    "UnsupportedTermDiagnostic",
    "inspect_trajectory_problem_backend",
    "TrajectoryIocCompiledProblem",
    "compile_trajectory_ioc_problem",
    "estimate_ioc_weights",
    "problem_adapter",
    "vision",
    "kots",
    "pinocchio",
]
