from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.state_cache import StateCache
from ..core.time_grid import TimeGrid
from ..expr.registry import Registry
from ..model.term import VariablePack


@dataclass
class BuilderContext:
    pack: VariablePack
    state_cache: StateCache
    time: TimeGrid
    registry: Registry
    model: Any = None
