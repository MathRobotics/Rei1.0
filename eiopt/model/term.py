from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple, List, Callable, Any, Iterable

import numpy as np

from ..core.state_cache import StateKey

Array = np.ndarray


@dataclass
class Variable:
    name: str
    x: Array  # shape (n,)

    def dim(self) -> int:
        return int(np.asarray(self.x).size)


def pack(vars: Sequence[Variable]) -> Array:
    if len(vars) == 0:
        return np.zeros((0,), dtype=float)
    return np.concatenate([np.asarray(v.x).reshape(-1) for v in vars], axis=0)


def total_dim(vars: Sequence[Variable]) -> int:
    return int(sum(v.dim() for v in vars))


@dataclass
class VariablePack:
    """Global variable ordering (the only source of Jacobian column order)."""

    vars: Sequence[Variable]
    revision: int = 0

    def __post_init__(self) -> None:
        names = [v.name for v in self.vars]
        if len(names) != len(set(names)):
            raise ValueError(f"VariablePack: duplicate variable names: {names}")

        self.slices: dict[str, Tuple[int, int]] = {}
        col = 0
        for v in self.vars:
            n = v.dim()
            self.slices[v.name] = (col, col + n)
            col += n
        self.n_total = int(col)

    def get(self) -> Array:
        return pack(self.vars)

    def apply_dx(self, dx: Array) -> None:
        dx = np.asarray(dx, dtype=float).reshape(-1)
        if dx.size != self.n_total:
            raise ValueError(f"apply_dx: expected {self.n_total}, got {dx.size}")

        for v in self.vars:
            s, e = self.slices[v.name]
            v.x = np.asarray(v.x, dtype=float).reshape(-1) + dx[s:e]


@dataclass(frozen=True)
class EvalContext:
    """Minimal evaluation context passed to Expr.eval()."""

    pack: VariablePack
    state: Any = None
    time: Any = None
    revision: int = 0


class Expr(Protocol):
    name: str
    vars: Sequence[Variable]

    def eval(self, ctx: EvalContext) -> Tuple[Array, Sequence[Array]]:
        """Return (residual, Jacobian blocks aligned with self.vars)."""

    def deps(self) -> Iterable[StateKey]:
        """Return StateKey dependencies used by this expression."""


@dataclass
class DirectVectorExpr:
    """Adapter for legacy callables.

    - fn_value(ctx) -> (m,)
    - fn_blocks(ctx) -> list[blocks], aligned with vars
    """

    name: str
    vars: Sequence[Variable]
    fn_value: Callable[[EvalContext], Array]
    fn_blocks: Callable[[EvalContext], Sequence[Array]]

    def eval(self, ctx: EvalContext) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(self.fn_value(ctx), dtype=float).reshape(-1)

        blocks = [np.asarray(B, dtype=float) for B in self.fn_blocks(ctx)]
        if len(blocks) != len(self.vars):
            raise ValueError(f"{self.name}: len(blocks) != len(vars): {len(blocks)} vs {len(self.vars)}")

        m = int(r.size)
        checked: List[Array] = []
        for v, B in zip(self.vars, blocks):
            if B.shape != (m, v.dim()):
                raise ValueError(
                    f"{self.name}: block shape mismatch for var '{v.name}'. "
                    f"expected {(m, v.dim())}, got {B.shape}"
                )
            checked.append(B)
        return r, checked

    def deps(self) -> Iterable[StateKey]:
        return []


class Cost(Protocol):
    name: str

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]: ...


@dataclass
class L2Cost:
    name: str = "l2"

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(r, dtype=float).reshape(-1)
        blocks2 = [np.asarray(B, dtype=float) for B in blocks]
        return r, blocks2


@dataclass
class DiagonalWeightCost:
    w: Array  # (m,)
    name: str = "diag_weight"

    def __post_init__(self) -> None:
        self.w = np.asarray(self.w, dtype=float).reshape(-1)
        if np.any(self.w < 0):
            raise ValueError("DiagonalWeightCost: w must be >= 0.")

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(r, dtype=float).reshape(-1)
        if r.size != self.w.size:
            raise ValueError(f"DiagonalWeightCost: size mismatch. r={r.size}, w={self.w.size}")

        sw = np.sqrt(self.w)
        r2 = sw * r
        blocks2 = [sw[:, None] * np.asarray(B, dtype=float) for B in blocks]
        return r2, blocks2


@dataclass
class ScalarWeightCost:
    w: float
    name: str = "scalar_weight"

    def __post_init__(self) -> None:
        if self.w < 0:
            raise ValueError("ScalarWeightCost: w must be >= 0.")

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(r, dtype=float).reshape(-1)
        sw = float(np.sqrt(self.w))
        r2 = sw * r
        blocks2 = [sw * np.asarray(B, dtype=float) for B in blocks]
        return r2, blocks2


@dataclass
class HuberCost:
    delta: float
    name: str = "huber"

    def __post_init__(self) -> None:
        if self.delta <= 0:
            raise ValueError("HuberCost: delta must be > 0.")

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(r, dtype=float).reshape(-1)
        nr = float(np.linalg.norm(r)) + 1e-12
        w = 1.0 if nr <= self.delta else (self.delta / nr)
        sw = float(np.sqrt(w))
        r2 = sw * r
        blocks2 = [sw * np.asarray(B, dtype=float) for B in blocks]
        return r2, blocks2

