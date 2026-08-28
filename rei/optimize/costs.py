from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple

import numpy as np

Array = np.ndarray


class Cost(Protocol):
    name: str

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]: ...

    def residual_vjp(self, r: Array, rhs: Array) -> Array:
        """Map a weighted-residual cotangent back to the raw residual."""


@dataclass
class L2Cost:
    name: str = "l2"

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(r, dtype=float).reshape(-1)
        blocks2 = [np.asarray(B, dtype=float) for B in blocks]
        return r, blocks2

    def residual_vjp(self, r: Array, rhs: Array) -> Array:
        del r
        return np.asarray(rhs, dtype=float).copy()


@dataclass
class DiagonalWeightCost:
    w: Array  # (m,)
    name: str = "diag_weight"

    def __post_init__(self) -> None:
        self.set_weight(self.w)

    def set_weight(self, w: Array) -> None:
        w_arr = np.asarray(w, dtype=float).reshape(-1)
        if np.any(w_arr < 0):
            raise ValueError("DiagonalWeightCost: w must be >= 0.")
        self.w = w_arr

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(r, dtype=float).reshape(-1)
        if r.size != self.w.size:
            raise ValueError(f"DiagonalWeightCost: size mismatch. r={r.size}, w={self.w.size}")

        sw = np.sqrt(self.w)
        r2 = sw * r
        blocks2 = [sw[:, None] * np.asarray(B, dtype=float) for B in blocks]
        return r2, blocks2

    def residual_vjp(self, r: Array, rhs: Array) -> Array:
        r_vec = np.asarray(r, dtype=float).reshape(-1)
        R = np.asarray(rhs, dtype=float)
        if R.ndim not in (1, 2) or int(R.shape[0]) != r_vec.size:
            raise ValueError(
                "DiagonalWeightCost.residual_vjp: rhs must have leading dimension "
                f"{r_vec.size}, got {R.shape}."
            )
        scale = np.sqrt(self.w)
        return scale * R if R.ndim == 1 else scale.reshape(-1, 1) * R


@dataclass
class ScalarWeightCost:
    w: float
    name: str = "scalar_weight"

    def __post_init__(self) -> None:
        self.set_weight(self.w)

    def set_weight(self, w: float) -> None:
        w_f = float(w)
        if w_f < 0:
            raise ValueError("ScalarWeightCost: w must be >= 0.")
        self.w = w_f

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(r, dtype=float).reshape(-1)
        sw = float(np.sqrt(self.w))
        r2 = sw * r
        blocks2 = [sw * np.asarray(B, dtype=float) for B in blocks]
        return r2, blocks2

    def residual_vjp(self, r: Array, rhs: Array) -> Array:
        del r
        return float(np.sqrt(self.w)) * np.asarray(rhs, dtype=float)


@dataclass
class HuberCost:
    delta: float
    name: str = "huber"

    def __post_init__(self) -> None:
        if self.delta <= 0:
            raise ValueError("HuberCost: delta must be > 0.")

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(r, dtype=float).reshape(-1)
        sw = self._scale(r)
        r2 = sw * r
        blocks2 = [sw * np.asarray(B, dtype=float) for B in blocks]
        return r2, blocks2

    def _scale(self, r: Array) -> float:
        nr = float(np.linalg.norm(np.asarray(r, dtype=float).reshape(-1))) + 1e-12
        return float(np.sqrt(1.0 if nr <= self.delta else (self.delta / nr)))

    def residual_vjp(self, r: Array, rhs: Array) -> Array:
        # Match apply(): this is the IRLS/Gauss-Newton weighting used by Rei,
        # not the derivative of the scale itself.
        return self._scale(r) * np.asarray(rhs, dtype=float)

__all__ = [
    "Array",
    "Cost",
    "L2Cost",
    "DiagonalWeightCost",
    "ScalarWeightCost",
    "HuberCost",
]
