"""Optional execution plans for sharing expression derivative products."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np


class TermProductPlan(Protocol):
    """Return accelerated results; omitted terms use normal expression evaluation."""

    def shared_jvp(self, runtime: Any, indices: Iterable[int], direction: np.ndarray) -> dict[int, np.ndarray]: ...

    def residual_vjp(
        self, runtime: Any, *, terms: list[Any], raw_rhs_by_index: dict[int, np.ndarray],
    ) -> tuple[dict[int, np.ndarray], np.ndarray, set[int]]: ...

    def gradient_contributions(
        self, runtime: Any, *, term_indices: Iterable[int],
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]: ...
