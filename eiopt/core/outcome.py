from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .timing import TimingReport

Array = np.ndarray


@dataclass(frozen=True)
class SolveStats:
    status: str
    iterations: int
    initial_objective: float | None = None
    objective: float | None = None
    residual_norm: float | None = None
    step_norm: float | None = None
    message: str = ""

    @property
    def converged(self) -> bool:
        return str(self.status).strip().lower() == "converged"

    def to_summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": str(self.status),
            "converged": bool(self.converged),
            "iters": int(self.iterations),
        }
        if self.initial_objective is not None:
            out["cost0"] = float(self.initial_objective)
        if self.objective is not None:
            out["cost"] = float(self.objective)
        if self.residual_norm is not None:
            out["rnorm"] = float(self.residual_norm)
        if self.step_norm is not None:
            out["dxnorm"] = float(self.step_norm)
        if str(self.message) != "":
            out["message"] = str(self.message)
        return out


@dataclass
class SolveOutcome:
    solution: Array
    stats: SolveStats
    timing: TimingReport
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.solution = np.asarray(self.solution, dtype=float).reshape(-1).copy()
        self.meta = dict(self.meta)

    @property
    def converged(self) -> bool:
        return bool(self.stats.converged)

    @property
    def status(self) -> str:
        return str(self.stats.status)

    @property
    def iterations(self) -> int:
        return int(self.stats.iterations)

    def to_summary(self) -> dict[str, Any]:
        return self.stats.to_summary()


__all__ = [
    "Array",
    "SolveStats",
    "SolveOutcome",
]
