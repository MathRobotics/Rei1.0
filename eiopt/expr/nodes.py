from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.state_cache import StateKey
from ..model.term import Variable, RuntimeContext, Expr


@dataclass
class GetStateExpr:
    name: str
    vars: Sequence[Variable]
    key_value: StateKey
    key_jac_q: StateKey

    def deps(self):
        return [self.key_value, self.key_jac_q]

    def eval(self, ctx: RuntimeContext):
        sc = ctx.state  # StateCache-like
        y = np.asarray(sc.get(self.key_value), dtype=float).reshape(-1)
        J = np.asarray(sc.get(self.key_jac_q), dtype=float)
        if J.shape != (y.size, self.vars[0].dim()):
            raise ValueError(f"{self.name}: J shape mismatch: {J.shape} vs {(y.size, self.vars[0].dim())}")
        return y, [J]


@dataclass
class GetVarExpr:
    """Read decision variables directly from the VariablePack (no StateCache deps).

    If `k` is provided and the variable dimension is divisible by (time.N+1),
    the variable is treated as a stacked trajectory and sliced by k.
    """

    name: str
    vars: Sequence[Variable]
    k: int | None = None

    def deps(self):
        return []

    def eval(self, ctx: RuntimeContext):
        x = np.asarray(self.vars[0].x, dtype=float).reshape(-1)
        n_total = int(x.size)

        if self.k is None:
            return x.copy(), [np.eye(n_total, dtype=float)]

        k = int(self.k)

        steps = 1
        time = getattr(ctx, "time", None)
        if time is not None and hasattr(time, "N"):
            try:
                steps = int(time.N) + 1
            except Exception:
                steps = 1

        chunked = bool(steps > 1 and n_total % steps == 0)
        if not chunked:
            if k != 0:
                raise ValueError(
                    f"{self.name}: requested k={k}, but variable '{self.vars[0].name}' is not time-chunked "
                    f"(dim={n_total}, steps={steps})."
                )
            return x.copy(), [np.eye(n_total, dtype=float)]

        if k < 0 or k >= steps:
            raise ValueError(f"{self.name}: requested k={k}, but time steps are 0..{steps - 1}.")

        n = int(n_total // steps)
        start = int(k * n)
        y = x[start : start + n].copy()

        J = np.zeros((n, n_total), dtype=float)
        J[:, start : start + n] = np.eye(n, dtype=float)
        return y, [J]


@dataclass
class ConstantExpr:
    name: str
    value: np.ndarray
    vars: Sequence[Variable] = ()

    def deps(self):
        return []

    def eval(self, ctx: RuntimeContext):
        y = np.asarray(self.value, dtype=float).reshape(-1)
        blocks = [np.zeros((y.size, v.dim()), dtype=float) for v in self.vars]
        return y, blocks


@dataclass
class SubExpr:
    name: str
    a: Expr
    b: Expr

    @property
    def vars(self):
        return self.a.vars

    def deps(self):
        return list(self.a.deps()) + list(self.b.deps())

    def eval(self, ctx: RuntimeContext):
        ra, Ja = self.a.eval(ctx)
        rb, Jb = self.b.eval(ctx)
        if ra.shape != rb.shape:
            raise ValueError(f"{self.name}: shape mismatch {ra.shape} vs {rb.shape}")
        if len(Ja) != len(Jb):
            raise ValueError(f"{self.name}: block len mismatch {len(Ja)} vs {len(Jb)}")
        r = ra - rb
        blocks = [A - B for A, B in zip(Ja, Jb)]
        return r, blocks


@dataclass
class StackExpr:
    name: str
    parts: Sequence[Expr]

    @property
    def vars(self):
        return self.parts[0].vars if self.parts else []

    def deps(self):
        out = []
        for p in self.parts:
            out.extend(list(p.deps()))
        return out

    def eval(self, ctx: RuntimeContext):
        r_list = []
        J_list = None
        for p in self.parts:
            r, blocks = p.eval(ctx)
            r_list.append(np.asarray(r, float).reshape(-1))
            if J_list is None:
                J_list = [[] for _ in blocks]
            for i, B in enumerate(blocks):
                J_list[i].append(np.asarray(B, float))
        r_all = np.concatenate(r_list, axis=0) if r_list else np.zeros((0,), float)
        blocks_all = [np.vstack(chunks) for chunks in (J_list or [])]
        return r_all, blocks_all


@dataclass
class HingeExpr:
    name: str
    base: Expr

    @property
    def vars(self):
        return self.base.vars

    def deps(self):
        return self.base.deps()

    def eval(self, ctx: RuntimeContext):
        h, blocks = self.base.eval(ctx)

        h = np.asarray(h, dtype=float).reshape(-1)
        m = h.size

        active = (h > 0.0).astype(float)
        r = np.maximum(0.0, h)

        blocks2 = []
        for B in blocks:
            B = np.asarray(B, dtype=float)
            if B.shape[0] != m:
                raise ValueError(f"{self.name}: block row mismatch: h has {m}, block has {B.shape}")
            blocks2.append(active[:, None] * B)
        return r, blocks2
