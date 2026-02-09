from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TimeGrid:
    N: int
    dt: float
    revision: int = 0

    def t(self, k: int) -> float:
        return float(k) * float(self.dt)

    def ks(self) -> range:
        return range(self.N + 1)

    def update(self, *, N: int | None = None, dt: float | None = None) -> None:
        changed = False
        if N is not None and int(N) != int(self.N):
            self.N = int(N)
            changed = True
        if dt is not None and float(dt) != float(self.dt):
            self.dt = float(dt)
            changed = True
        if changed:
            self.revision += 1

    @classmethod
    def single_time(cls) -> "TimeGrid":
        return cls(N=0, dt=0.0, revision=0)

    @classmethod
    def from_dsl(cls, dsl: dict) -> "TimeGrid":
        if dsl is None:
            raise ValueError("TimeGrid dsl is required")
        if "N" not in dsl or "dt" not in dsl:
            raise ValueError("TimeGrid dsl must contain 'N' and 'dt'")
        return cls(N=int(dsl["N"]), dt=float(dsl["dt"]))
