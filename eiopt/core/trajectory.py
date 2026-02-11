from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .bspline import bspline_basis_matrix, default_clamped_uniform_knots

Array = np.ndarray


@dataclass
class TrajectoryMap:
    """Affine map from trajectory parameters `p` to stacked generalized coordinates.

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
            raise ValueError(f"TrajectoryMap: A must be 2D, got shape {self.A.shape}.")
        if self.steps <= 0:
            raise ValueError(f"TrajectoryMap: steps must be > 0, got {self.steps}.")
        if self.q_dim <= 0:
            raise ValueError(f"TrajectoryMap: q_dim must be > 0, got {self.q_dim}.")

        traj_dim = int(self.steps * self.q_dim)
        if self.A.shape[0] != traj_dim:
            raise ValueError(
                "TrajectoryMap: A row mismatch. "
                f"Expected {traj_dim} (=steps*q_dim), got {self.A.shape[0]}."
            )
        if self.b.size != traj_dim:
            raise ValueError(
                "TrajectoryMap: b size mismatch. "
                f"Expected {traj_dim} (=steps*q_dim), got {self.b.size}."
            )

    @property
    def p_dim(self) -> int:
        return int(self.A.shape[1])

    def _row_slice(self, k: int) -> slice:
        k = int(k)
        if k < 0 or k >= self.steps:
            raise ValueError(f"TrajectoryMap: k must be in 0..{self.steps - 1}, got {k}.")
        start = int(k * self.q_dim)
        return slice(start, start + self.q_dim)

    def q_at(self, p: Array, k: int) -> Array:
        p_vec = np.asarray(p, dtype=float).reshape(-1)
        if p_vec.size != self.p_dim:
            raise ValueError(f"TrajectoryMap: p size mismatch. Expected {self.p_dim}, got {p_vec.size}.")
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
    ) -> "TrajectoryMap":
        if len(A_blocks) == 0:
            raise ValueError("TrajectoryMap.from_blocks: A_blocks must be non-empty.")

        A_mats = [np.asarray(Ak, dtype=float) for Ak in A_blocks]
        if any(Ak.ndim != 2 for Ak in A_mats):
            bad = [Ak.shape for Ak in A_mats if Ak.ndim != 2]
            raise ValueError(f"TrajectoryMap.from_blocks: all A_blocks must be 2D, got {bad}.")

        q_dim = int(A_mats[0].shape[0])
        p_dim = int(A_mats[0].shape[1])
        if q_dim <= 0 or p_dim <= 0:
            raise ValueError(f"TrajectoryMap.from_blocks: invalid first block shape {A_mats[0].shape}.")

        for i, Ak in enumerate(A_mats):
            if Ak.shape != (q_dim, p_dim):
                raise ValueError(
                    "TrajectoryMap.from_blocks: block shape mismatch. "
                    f"A_blocks[{i}] has {Ak.shape}, expected {(q_dim, p_dim)}."
                )

        if b_blocks is None:
            b_vec = np.zeros((len(A_mats) * q_dim,), dtype=float)
        else:
            if len(b_blocks) != len(A_mats):
                raise ValueError(
                    "TrajectoryMap.from_blocks: len(b_blocks) must match len(A_blocks). "
                    f"Got {len(b_blocks)} vs {len(A_mats)}."
                )
            b_parts = [np.asarray(bk, dtype=float).reshape(-1) for bk in b_blocks]
            for i, bk in enumerate(b_parts):
                if bk.size != q_dim:
                    raise ValueError(
                        "TrajectoryMap.from_blocks: b block size mismatch. "
                        f"b_blocks[{i}] has size {bk.size}, expected {q_dim}."
                    )
            b_vec = np.concatenate(b_parts, axis=0)

        A_all = np.vstack(A_mats)
        return cls(A=A_all, b=b_vec, steps=len(A_mats), q_dim=q_dim)

    @classmethod
    def from_bspline(
        cls,
        *,
        steps: int,
        q_dim: int,
        degree: int,
        num_ctrl_points: int,
        knot_vector: Array | None = None,
        u_samples: Array | None = None,
    ) -> "TrajectoryMap":
        """Build a clamped B-spline trajectory map.

        Decision variable `p` is stacked control points:

          p = [c0, c1, ..., c_{M-1}],  ci in R^q_dim

        and the trajectory is sampled at `steps` parameter values:

          q(k) = sum_i N_i(u_k) * c_i
        """

        steps = int(steps)
        q_dim = int(q_dim)
        degree = int(degree)
        num_ctrl_points = int(num_ctrl_points)

        if steps <= 0:
            raise ValueError(f"TrajectoryMap.from_bspline: steps must be > 0, got {steps}.")
        if q_dim <= 0:
            raise ValueError(f"TrajectoryMap.from_bspline: q_dim must be > 0, got {q_dim}.")
        if degree < 0:
            raise ValueError(f"TrajectoryMap.from_bspline: degree must be >= 0, got {degree}.")
        if num_ctrl_points < degree + 1:
            raise ValueError(
                "TrajectoryMap.from_bspline: num_ctrl_points must satisfy "
                f"num_ctrl_points >= degree + 1, got {num_ctrl_points} and {degree}."
            )

        if knot_vector is None:
            knots = default_clamped_uniform_knots(
                num_ctrl_points=num_ctrl_points,
                degree=degree,
            )
        else:
            knots = np.asarray(knot_vector, dtype=float).reshape(-1)

        expected_knot_size = int(num_ctrl_points + degree + 1)
        if knots.size != expected_knot_size:
            raise ValueError(
                "TrajectoryMap.from_bspline: knot vector size mismatch. "
                f"Expected {expected_knot_size}, got {knots.size}."
            )
        if np.any(np.diff(knots) < 0.0):
            raise ValueError("TrajectoryMap.from_bspline: knot vector must be non-decreasing.")
        if not np.allclose(knots[: degree + 1], knots[0], atol=1e-12, rtol=0.0):
            raise ValueError(
                "TrajectoryMap.from_bspline: knot vector must be clamped at start "
                f"(first {degree + 1} knots equal)."
            )
        if not np.allclose(knots[-(degree + 1) :], knots[-1], atol=1e-12, rtol=0.0):
            raise ValueError(
                "TrajectoryMap.from_bspline: knot vector must be clamped at end "
                f"(last {degree + 1} knots equal)."
            )

        u_min = float(knots[degree])
        u_max = float(knots[num_ctrl_points])
        if u_max <= u_min:
            raise ValueError(
                "TrajectoryMap.from_bspline: invalid knot domain. "
                f"Expected knots[degree] < knots[num_ctrl_points], got {u_min} >= {u_max}."
            )

        if u_samples is None:
            u_vec = np.linspace(u_min, u_max, steps, dtype=float)
        else:
            u_vec = np.asarray(u_samples, dtype=float).reshape(-1)
            if u_vec.size != steps:
                raise ValueError(
                    "TrajectoryMap.from_bspline: u_samples size mismatch. "
                    f"Expected {steps}, got {u_vec.size}."
                )

        tol = 1e-12
        if np.any(u_vec < (u_min - tol)) or np.any(u_vec > (u_max + tol)):
            raise ValueError(
                "TrajectoryMap.from_bspline: u_samples must lie in spline domain "
                f"[{u_min}, {u_max}]."
            )
        u_vec = np.clip(u_vec, u_min, u_max)

        basis = bspline_basis_matrix(
            u_vec=u_vec,
            degree=degree,
            knots=knots,
            num_ctrl_points=num_ctrl_points,
        )
        eye = np.eye(q_dim, dtype=float)
        A = np.zeros((steps * q_dim, num_ctrl_points * q_dim), dtype=float)
        for k in range(steps):
            row = slice(k * q_dim, (k + 1) * q_dim)
            A[row, :] = np.kron(basis[k, :], eye)

        b = np.zeros((steps * q_dim,), dtype=float)
        return cls(A=A, b=b, steps=steps, q_dim=q_dim)
