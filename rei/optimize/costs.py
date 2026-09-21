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
        if not np.all(np.isfinite(w_arr)) or np.any(w_arr < 0):
            raise ValueError("DiagonalWeightCost: w must be finite and >= 0.")
        self.w = w_arr.copy()

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
        if not np.isfinite(w_f) or w_f < 0:
            raise ValueError("ScalarWeightCost: w must be finite and >= 0.")
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
    """Radial Huber loss represented by an exactly differentiated residual.

    Its squared norm is n**2 for n <= delta and 2*delta*n-delta**2
    otherwise, where n is the norm of the entire term's raw residual.
    """
    delta: float
    name: str = "huber"

    def __post_init__(self) -> None:
        if not np.isfinite(self.delta) or self.delta <= 0:
            raise ValueError("HuberCost: delta must be > 0.")

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(r, dtype=float).reshape(-1)
        sw = self._scale(r)
        r2 = sw * r
        blocks2 = [self.residual_vjp(r, B) for B in blocks]
        return r2, blocks2

    def _scale(self, r: Array) -> float:
        nr = float(np.linalg.norm(np.asarray(r, dtype=float).reshape(-1)))
        if nr <= self.delta:
            return 1.0
        ratio = self.delta / nr
        return float(np.sqrt(ratio * (2.0 - ratio)))

    def residual_vjp(self, r: Array, rhs: Array) -> Array:
        r = np.asarray(r, dtype=float).reshape(-1)
        R = np.asarray(rhs, dtype=float)
        if R.ndim not in (1, 2) or R.shape[0] != r.size:
            raise ValueError("HuberCost.residual_vjp: rhs must match the raw residual rows.")
        nr = float(np.linalg.norm(r))
        if nr <= self.delta:
            return R.copy()
        scale = self._scale(r)
        unit = r / nr
        radial = self.delta / (nr * scale) - scale
        projected = unit * (unit @ R) if R.ndim == 1 else unit[:, None] * (unit @ R)
        # The residual transformation has a symmetric Jacobian, so this
        # also applies its derivative to forward Jacobian blocks in apply().
        return scale * R + radial * projected

__all__ = [
    "Array",
    "Cost",
    "L2Cost",
    "DiagonalWeightCost",
    "ScalarWeightCost",
    "HuberCost",
]
