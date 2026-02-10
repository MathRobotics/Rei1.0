from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass
class LinearTrajectoryMap:
    """Linear map from trajectory parameters `p` to stacked generalized coordinates.

    The trajectory is represented as:

      q_traj = A @ p + b

    where `q_traj` stacks q(k) for k=0..steps-1.
    """

    A: Array
    b: Array
    steps: int
    q_dim: int

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.b = np.asarray(self.b, dtype=float).reshape(-1)
        self.steps = int(self.steps)
        self.q_dim = int(self.q_dim)

        if self.A.ndim != 2:
            raise ValueError(f"LinearTrajectoryMap: A must be 2D, got shape {self.A.shape}.")
        if self.steps <= 0:
            raise ValueError(f"LinearTrajectoryMap: steps must be > 0, got {self.steps}.")
        if self.q_dim <= 0:
            raise ValueError(f"LinearTrajectoryMap: q_dim must be > 0, got {self.q_dim}.")

        traj_dim = int(self.steps * self.q_dim)
        if self.A.shape[0] != traj_dim:
            raise ValueError(
                "LinearTrajectoryMap: A row mismatch. "
                f"Expected {traj_dim} (=steps*q_dim), got {self.A.shape[0]}."
            )
        if self.b.size != traj_dim:
            raise ValueError(
                "LinearTrajectoryMap: b size mismatch. "
                f"Expected {traj_dim} (=steps*q_dim), got {self.b.size}."
            )

    @property
    def p_dim(self) -> int:
        return int(self.A.shape[1])

    def _row_slice(self, k: int) -> slice:
        k = int(k)
        if k < 0 or k >= self.steps:
            raise ValueError(f"LinearTrajectoryMap: k must be in 0..{self.steps - 1}, got {k}.")
        start = int(k * self.q_dim)
        return slice(start, start + self.q_dim)

    def q_at(self, p: Array, k: int) -> Array:
        p_vec = np.asarray(p, dtype=float).reshape(-1)
        if p_vec.size != self.p_dim:
            raise ValueError(f"LinearTrajectoryMap: p size mismatch. Expected {self.p_dim}, got {p_vec.size}.")
        s = self._row_slice(k)
        return (self.A[s, :] @ p_vec + self.b[s]).reshape(-1)

    def dqdp_at(self, k: int) -> Array:
        s = self._row_slice(k)
        return self.A[s, :].copy()

    @classmethod
    def from_blocks(
        cls,
        A_blocks: Sequence[Array],
        *,
        b_blocks: Sequence[Array] | None = None,
    ) -> "LinearTrajectoryMap":
        if len(A_blocks) == 0:
            raise ValueError("LinearTrajectoryMap.from_blocks: A_blocks must be non-empty.")

        A_mats = [np.asarray(Ak, dtype=float) for Ak in A_blocks]
        if any(Ak.ndim != 2 for Ak in A_mats):
            bad = [Ak.shape for Ak in A_mats if Ak.ndim != 2]
            raise ValueError(f"LinearTrajectoryMap.from_blocks: all A_blocks must be 2D, got {bad}.")

        q_dim = int(A_mats[0].shape[0])
        p_dim = int(A_mats[0].shape[1])
        if q_dim <= 0 or p_dim <= 0:
            raise ValueError(f"LinearTrajectoryMap.from_blocks: invalid first block shape {A_mats[0].shape}.")

        for i, Ak in enumerate(A_mats):
            if Ak.shape != (q_dim, p_dim):
                raise ValueError(
                    "LinearTrajectoryMap.from_blocks: block shape mismatch. "
                    f"A_blocks[{i}] has {Ak.shape}, expected {(q_dim, p_dim)}."
                )

        if b_blocks is None:
            b_vec = np.zeros((len(A_mats) * q_dim,), dtype=float)
        else:
            if len(b_blocks) != len(A_mats):
                raise ValueError(
                    "LinearTrajectoryMap.from_blocks: len(b_blocks) must match len(A_blocks). "
                    f"Got {len(b_blocks)} vs {len(A_mats)}."
                )
            b_parts = [np.asarray(bk, dtype=float).reshape(-1) for bk in b_blocks]
            for i, bk in enumerate(b_parts):
                if bk.size != q_dim:
                    raise ValueError(
                        "LinearTrajectoryMap.from_blocks: b block size mismatch. "
                        f"b_blocks[{i}] has size {bk.size}, expected {q_dim}."
                    )
            b_vec = np.concatenate(b_parts, axis=0)

        A_all = np.vstack(A_mats)
        return cls(A=A_all, b=b_vec, steps=len(A_mats), q_dim=q_dim)
