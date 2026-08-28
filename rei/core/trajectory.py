from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .bspline import bspline_basis_derivative_matrices, default_clamped_uniform_knots

Array = np.ndarray


@dataclass(frozen=True)
class BsplineTrajectoryOperator:
    """Block-sparse ``kron(basis, I_q)`` trajectory linear operator.

    The basis has shape ``(steps, num_ctrl_points)``.  Unlike its equivalent
    Kronecker matrix, this representation stores only the scalar B-spline
    basis and applies it directly to control-point blocks.
    """

    basis: Array
    q_dim: int

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=float)
        if basis.ndim != 2:
            raise ValueError(f"BsplineTrajectoryOperator: basis must be 2D, got {basis.shape}.")
        if int(self.q_dim) <= 0:
            raise ValueError("BsplineTrajectoryOperator: q_dim must be positive.")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "q_dim", int(self.q_dim))

    @property
    def steps(self) -> int:
        return int(self.basis.shape[0])

    @property
    def num_ctrl_points(self) -> int:
        return int(self.basis.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.steps * self.q_dim), int(self.num_ctrl_points * self.q_dim))

    @property
    def ndim(self) -> int:
        return 2

    @property
    def T(self) -> "BsplineTrajectoryTransposeOperator":
        return BsplineTrajectoryTransposeOperator(self)

    def matvec(self, p: Array) -> Array:
        p_vec = np.asarray(p, dtype=float).reshape(-1)
        if p_vec.size != self.shape[1]:
            raise ValueError(f"BsplineTrajectoryOperator: p size mismatch. Expected {self.shape[1]}, got {p_vec.size}.")
        controls = p_vec.reshape(self.num_ctrl_points, self.q_dim)
        return np.asarray(self.basis @ controls, dtype=float).reshape(-1)

    def matvec_at(self, p: Array, k: int) -> Array:
        k_i = int(k)
        if k_i < 0 or k_i >= self.steps:
            raise ValueError(f"BsplineTrajectoryOperator: k must be in 0..{self.steps - 1}, got {k_i}.")
        p_vec = np.asarray(p, dtype=float).reshape(-1)
        if p_vec.size != self.shape[1]:
            raise ValueError(f"BsplineTrajectoryOperator: p size mismatch. Expected {self.shape[1]}, got {p_vec.size}.")
        return np.asarray(self.basis[k_i] @ p_vec.reshape(self.num_ctrl_points, self.q_dim), dtype=float).reshape(-1)

    def rmatvec(self, rhs: Array) -> Array:
        r = np.asarray(rhs, dtype=float)
        if r.ndim == 1:
            if r.size != self.shape[0]:
                raise ValueError(f"BsplineTrajectoryOperator: rhs size mismatch. Expected {self.shape[0]}, got {r.size}.")
            return np.asarray(self.basis.T @ r.reshape(self.steps, self.q_dim), dtype=float).reshape(-1)
        if r.ndim == 2:
            if r.shape[0] != self.shape[0]:
                raise ValueError(f"BsplineTrajectoryOperator: rhs row mismatch. Expected {self.shape[0]}, got {r.shape[0]}.")
            values = r.reshape(self.steps, self.q_dim, r.shape[1])
            return np.einsum("tc,tqk->cqk", self.basis, values, optimize=True).reshape(self.shape[1], r.shape[1])
        raise ValueError(f"BsplineTrajectoryOperator: rhs must be 1D or 2D, got {r.shape}.")

    def rmatvec_at(self, k: int, rhs: Array) -> Array:
        k_i = int(k)
        r = np.asarray(rhs, dtype=float)
        if k_i < 0 or k_i >= self.steps:
            raise ValueError(f"BsplineTrajectoryOperator: k must be in 0..{self.steps - 1}, got {k_i}.")
        if r.ndim == 1:
            if r.size != self.q_dim:
                raise ValueError(f"BsplineTrajectoryOperator: rhs size mismatch. Expected {self.q_dim}, got {r.size}.")
            return np.outer(self.basis[k_i], r).reshape(-1)
        if r.ndim == 2:
            if r.shape[0] != self.q_dim:
                raise ValueError(f"BsplineTrajectoryOperator: rhs row mismatch. Expected {self.q_dim}, got {r.shape[0]}.")
            return np.einsum("c,qk->cqk", self.basis[k_i], r, optimize=True).reshape(self.shape[1], r.shape[1])
        raise ValueError(f"BsplineTrajectoryOperator: rhs must be 1D or 2D, got {r.shape}.")

    def to_dense(self) -> Array:
        return np.kron(self.basis, np.eye(self.q_dim, dtype=float))

    def __array__(self, dtype=None, copy=None) -> Array:
        out = self.to_dense()
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out.copy() if copy else out

    def __matmul__(self, other: Array) -> Array:
        values = np.asarray(other, dtype=float)
        if values.ndim == 1:
            return self.matvec(values)
        if values.ndim == 2:
            if values.shape[0] != self.shape[1]:
                raise ValueError(f"BsplineTrajectoryOperator: RHS shape mismatch {values.shape}.")
            controls = values.reshape(self.num_ctrl_points, self.q_dim, values.shape[1])
            return np.einsum("tc,cqk->tqk", self.basis, controls, optimize=True).reshape(self.shape[0], values.shape[1])
        return NotImplemented

    def __getitem__(self, key):
        # Compatibility path for callers that explicitly request dense rows.
        return self.to_dense()[key]

    def __mul__(self, scalar: float) -> Array:
        return self.to_dense() * scalar

    def __rmul__(self, scalar: float) -> Array:
        return scalar * self.to_dense()

    def copy(self) -> Array:
        """Return a dense compatibility copy; use matvec/rmatvec in runtime paths."""
        return self.to_dense().copy()


@dataclass(frozen=True)
class BsplineTrajectoryTransposeOperator:
    source: BsplineTrajectoryOperator

    @property
    def shape(self) -> tuple[int, int]:
        return self.source.shape[::-1]

    @property
    def ndim(self) -> int:
        return 2

    def __matmul__(self, rhs: Array) -> Array:
        return self.source.rmatvec(rhs)


@dataclass
class TrajectoryMap:
    """Affine map from trajectory parameters `p` to stacked generalized coordinates.

    The trajectory is represented as:

      q_traj = A @ p + b

    where `q_traj` stacks q(k) for k=0..steps-1.
    """

    A: Array | BsplineTrajectoryOperator
    b: Array
    steps: int
    q_dim: int

    def __post_init__(self) -> None:
        if not isinstance(self.A, BsplineTrajectoryOperator):
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

    @property
    def is_sparse_operator(self) -> bool:
        return isinstance(self.A, BsplineTrajectoryOperator)

    def apply(self, p: Array) -> Array:
        return np.asarray(self.A @ np.asarray(p, dtype=float).reshape(-1), dtype=float).reshape(-1) + self.b

    def apply_transpose(self, rhs: Array) -> Array:
        if isinstance(self.A, BsplineTrajectoryOperator):
            return self.A.rmatvec(rhs)
        return np.asarray(self.A.T @ np.asarray(rhs, dtype=float), dtype=float)

    def apply_at(self, p: Array, k: int) -> Array:
        if isinstance(self.A, BsplineTrajectoryOperator):
            return self.A.matvec_at(p, k) + self.b[self._row_slice(k)]
        s = self._row_slice(k)
        return np.asarray(self.A[s, :] @ np.asarray(p, dtype=float).reshape(-1) + self.b[s], dtype=float).reshape(-1)

    def apply_transpose_at(self, k: int, rhs: Array) -> Array:
        if isinstance(self.A, BsplineTrajectoryOperator):
            return self.A.rmatvec_at(k, rhs)
        s = self._row_slice(k)
        return np.asarray(self.A[s, :].T @ np.asarray(rhs, dtype=float), dtype=float)

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
        return self.apply_at(p_vec, k)

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
        return cls.from_bspline_derivatives(
            steps=steps,
            q_dim=q_dim,
            degree=degree,
            num_ctrl_points=num_ctrl_points,
            knot_vector=knot_vector,
            u_samples=u_samples,
            max_derivative_order=0,
            parameter_scale=1.0,
        )[0]

    @classmethod
    def from_bspline_derivative(
        cls,
        *,
        steps: int,
        q_dim: int,
        degree: int,
        num_ctrl_points: int,
        knot_vector: Array | None = None,
        u_samples: Array | None = None,
        derivative_order: int = 1,
        parameter_scale: float = 1.0,
    ) -> "TrajectoryMap":
        """Build a derivative map of a clamped B-spline trajectory.

        Returns an affine map for sampled derivative trajectory:

          d^r q / dξ^r = A @ p + b

        where ``r = derivative_order`` and ``ξ`` is the spline parameter.
        ``parameter_scale`` can convert from parameter derivative to another axis,
        e.g. ``dq/dt = (du/dt) * dq/du`` by passing ``parameter_scale=du/dt``.
        """

        derivative_order = int(derivative_order)
        if derivative_order < 0:
            raise ValueError(
                "TrajectoryMap.from_bspline_derivative: derivative_order must be >= 0, "
                f"got {derivative_order}."
            )
        return cls.from_bspline_derivatives(
            steps=steps,
            q_dim=q_dim,
            degree=degree,
            num_ctrl_points=num_ctrl_points,
            knot_vector=knot_vector,
            u_samples=u_samples,
            max_derivative_order=derivative_order,
            parameter_scale=parameter_scale,
        )[derivative_order]

    @classmethod
    def from_bspline_derivatives(
        cls,
        *,
        steps: int,
        q_dim: int,
        degree: int,
        num_ctrl_points: int,
        knot_vector: Array | None = None,
        u_samples: Array | None = None,
        max_derivative_order: int,
        parameter_scale: float = 1.0,
    ) -> list["TrajectoryMap"]:
        """Build derivative maps up to ``max_derivative_order`` for a clamped B-spline trajectory.

        Returns a list ``maps`` where:

          maps[r] : d^r q / dξ^r = A_r @ p + b_r

        for ``r = 0..max_derivative_order``.
        """

        steps = int(steps)
        q_dim = int(q_dim)
        degree = int(degree)
        num_ctrl_points = int(num_ctrl_points)
        max_derivative_order = int(max_derivative_order)

        if steps <= 0:
            raise ValueError(f"TrajectoryMap.from_bspline_derivatives: steps must be > 0, got {steps}.")
        if q_dim <= 0:
            raise ValueError(f"TrajectoryMap.from_bspline_derivatives: q_dim must be > 0, got {q_dim}.")
        if degree < 0:
            raise ValueError(f"TrajectoryMap.from_bspline_derivatives: degree must be >= 0, got {degree}.")
        if num_ctrl_points < degree + 1:
            raise ValueError(
                "TrajectoryMap.from_bspline_derivatives: num_ctrl_points must satisfy "
                f"num_ctrl_points >= degree + 1, got {num_ctrl_points} and {degree}."
            )
        if max_derivative_order < 0:
            raise ValueError(
                "TrajectoryMap.from_bspline_derivatives: max_derivative_order must be >= 0, "
                f"got {max_derivative_order}."
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
                "TrajectoryMap.from_bspline_derivatives: knot vector size mismatch. "
                f"Expected {expected_knot_size}, got {knots.size}."
            )
        if np.any(np.diff(knots) < 0.0):
            raise ValueError("TrajectoryMap.from_bspline_derivatives: knot vector must be non-decreasing.")
        if not np.allclose(knots[: degree + 1], knots[0], atol=1e-12, rtol=0.0):
            raise ValueError(
                "TrajectoryMap.from_bspline_derivatives: knot vector must be clamped at start "
                f"(first {degree + 1} knots equal)."
            )
        if not np.allclose(knots[-(degree + 1) :], knots[-1], atol=1e-12, rtol=0.0):
            raise ValueError(
                "TrajectoryMap.from_bspline_derivatives: knot vector must be clamped at end "
                f"(last {degree + 1} knots equal)."
            )

        u_min = float(knots[degree])
        u_max = float(knots[num_ctrl_points])
        if u_max <= u_min:
            raise ValueError(
                "TrajectoryMap.from_bspline_derivatives: invalid knot domain. "
                f"Expected knots[degree] < knots[num_ctrl_points], got {u_min} >= {u_max}."
            )

        if u_samples is None:
            u_vec = np.linspace(u_min, u_max, steps, dtype=float)
        else:
            u_vec = np.asarray(u_samples, dtype=float).reshape(-1)
            if u_vec.size != steps:
                raise ValueError(
                    "TrajectoryMap.from_bspline_derivatives: u_samples size mismatch. "
                    f"Expected {steps}, got {u_vec.size}."
                )

        tol = 1e-12
        if np.any(u_vec < (u_min - tol)) or np.any(u_vec > (u_max + tol)):
            raise ValueError(
                "TrajectoryMap.from_bspline_derivatives: u_samples must lie in spline domain "
                f"[{u_min}, {u_max}]."
            )
        u_vec = np.clip(u_vec, u_min, u_max)

        basis_all = bspline_basis_derivative_matrices(
            u_vec=u_vec,
            degree=degree,
            knots=knots,
            num_ctrl_points=num_ctrl_points,
            max_derivative_order=max_derivative_order,
        )

        maps: list[TrajectoryMap] = []
        eye = np.eye(q_dim, dtype=float)
        for order in range(max_derivative_order + 1):
            scale = float(parameter_scale) ** order
            A = BsplineTrajectoryOperator(scale * basis_all[order, :, :], q_dim=q_dim)
            b = np.zeros((steps * q_dim,), dtype=float)
            maps.append(cls(A=A, b=b, steps=steps, q_dim=q_dim))
        return maps
