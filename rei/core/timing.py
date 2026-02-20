from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class TimingSpan:
    name: str
    seconds: float
    count: int = 1


@dataclass(frozen=True)
class TimingReport:
    total_seconds: float
    spans: tuple[TimingSpan, ...]

    def get(self, name: str, default: float = 0.0) -> float:
        for span in self.spans:
            if span.name == name:
                return float(span.seconds)
        return float(default)


class Profiler:
    """Lightweight hierarchical timer with named span aggregation."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = bool(enabled)
        self._t0 = perf_counter()
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def add(self, name: str, seconds: float, *, count: int = 1) -> None:
        if not self._enabled:
            return
        key = str(name)
        sec = float(seconds)
        if sec < 0.0:
            sec = 0.0
        self._sums[key] = float(self._sums.get(key, 0.0) + sec)
        self._counts[key] = int(self._counts.get(key, 0) + int(count))

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        if not self._enabled:
            yield
            return
        t0 = perf_counter()
        try:
            yield
        finally:
            self.add(str(name), perf_counter() - t0)

    def snapshot(self) -> TimingReport:
        total = float(perf_counter() - self._t0) if self._enabled else 0.0
        spans = tuple(
            TimingSpan(name=k, seconds=float(v), count=int(self._counts.get(k, 0)))
            for k, v in self._sums.items()
        )
        return TimingReport(total_seconds=total, spans=spans)


def ensure_profiler(profiler: Profiler | None) -> Profiler:
    if profiler is None:
        return Profiler(enabled=True)
    return profiler


__all__ = [
    "TimingSpan",
    "TimingReport",
    "Profiler",
    "ensure_profiler",
]
