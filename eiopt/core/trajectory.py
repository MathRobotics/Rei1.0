from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

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
            knots = cls._default_clamped_uniform_knots(
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

        basis = cls._bspline_basis_matrix(
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

    @classmethod
    def from_dsl(
        cls,
        dsl: Mapping[str, Any],
        *,
        default_steps: int | None = None,
        default_q_dim: int | None = None,
    ) -> "TrajectoryMap":
        """Build a trajectory map from DSL `trajectory` section.

        Supported forms:
          - `type = "bspline"` with keys:
              degree, num_ctrl_points, [knot_vector], [u_samples], [steps], [q_dim]
          - `type = "linear"` with keys:
              A, [b], [steps], [q_dim]

        `steps` / `q_dim` can be omitted when defaults are supplied.
        """

        if not isinstance(dsl, Mapping):
            raise TypeError("TrajectoryMap.from_dsl: dsl must be a mapping.")

        typ = str(dsl.get("type", "")).strip().lower()
        if typ == "":
            raise ValueError("TrajectoryMap.from_dsl: trajectory.type is required.")

        steps = cls._resolve_optional_positive_int(
            cls._pick_dsl_value(dsl, section=typ, key="steps"),
            name="steps",
            fallback=default_steps,
        )
        q_dim = cls._resolve_optional_positive_int(
            cls._pick_dsl_value(dsl, section=typ, key="q_dim"),
            name="q_dim",
            fallback=default_q_dim,
        )

        if typ == "bspline":
            if steps is None:
                raise ValueError(
                    "TrajectoryMap.from_dsl: steps is required for bspline trajectory "
                    "(set trajectory.steps or pass default_steps)."
                )
            if q_dim is None:
                raise ValueError(
                    "TrajectoryMap.from_dsl: q_dim is required for bspline trajectory "
                    "(set trajectory.q_dim or pass default_q_dim)."
                )

            degree = cls._resolve_required_nonnegative_int(
                cls._pick_dsl_value(dsl, section="bspline", key="degree"),
                name="degree",
            )
            num_ctrl_points = cls._resolve_required_positive_int(
                cls._pick_dsl_value(dsl, section="bspline", key="num_ctrl_points"),
                name="num_ctrl_points",
            )
            knot_vector_raw = cls._pick_dsl_value(dsl, section="bspline", key="knot_vector")
            u_samples_raw = cls._pick_dsl_value(dsl, section="bspline", key="u_samples")
            knot_vector = None if knot_vector_raw is None else np.asarray(knot_vector_raw, dtype=float).reshape(-1)
            u_samples = None if u_samples_raw is None else np.asarray(u_samples_raw, dtype=float).reshape(-1)
            return cls.from_bspline(
                steps=steps,
                q_dim=q_dim,
                degree=degree,
                num_ctrl_points=num_ctrl_points,
                knot_vector=knot_vector,
                u_samples=u_samples,
            )

        if typ == "linear":
            A_raw = cls._pick_dsl_value(dsl, section="linear", key="A")
            if A_raw is None:
                raise ValueError("TrajectoryMap.from_dsl: trajectory.linear.A is required for type='linear'.")
            try:
                A_arr = np.asarray(A_raw, dtype=float)
            except Exception as e:
                raise ValueError("TrajectoryMap.from_dsl: failed to parse linear A as numeric array.") from e

            if A_arr.ndim == 1:
                if steps is None or q_dim is None:
                    raise ValueError(
                        "TrajectoryMap.from_dsl: steps and q_dim are required when linear A is 1D "
                        "(flattened array)."
                    )
                rows = int(steps * q_dim)
                if rows <= 0:
                    raise ValueError("TrajectoryMap.from_dsl: invalid steps*q_dim for linear A reshape.")
                if A_arr.size % rows != 0:
                    raise ValueError(
                        "TrajectoryMap.from_dsl: linear A size mismatch. "
                        f"Expected multiple of {rows} (=steps*q_dim), got {A_arr.size}."
                    )
                A_mat = A_arr.reshape(rows, -1)
            elif A_arr.ndim == 2:
                A_mat = A_arr
            else:
                raise ValueError(
                    "TrajectoryMap.from_dsl: linear A must be 1D(flat) or 2D(matrix), "
                    f"got ndim={A_arr.ndim}."
                )

            rows = int(A_mat.shape[0])
            if steps is None and q_dim is None:
                raise ValueError(
                    "TrajectoryMap.from_dsl: cannot infer both steps and q_dim from linear A only. "
                    "Provide trajectory.steps or trajectory.q_dim (or defaults)."
                )
            if steps is None:
                if q_dim is None or rows % q_dim != 0:
                    raise ValueError(
                        "TrajectoryMap.from_dsl: failed to infer steps from linear A rows and q_dim. "
                        f"rows={rows}, q_dim={q_dim}."
                    )
                steps = int(rows // q_dim)
            if q_dim is None:
                if steps <= 0 or rows % steps != 0:
                    raise ValueError(
                        "TrajectoryMap.from_dsl: failed to infer q_dim from linear A rows and steps. "
                        f"rows={rows}, steps={steps}."
                    )
                q_dim = int(rows // steps)
            if int(steps * q_dim) != rows:
                raise ValueError(
                    "TrajectoryMap.from_dsl: linear A row mismatch against steps and q_dim. "
                    f"rows={rows}, steps*q_dim={steps * q_dim}."
                )

            b_raw = cls._pick_dsl_value(dsl, section="linear", key="b")
            if b_raw is None:
                b_vec = np.zeros((rows,), dtype=float)
            else:
                b_vec = np.asarray(b_raw, dtype=float).reshape(-1)
                if b_vec.size != rows:
                    raise ValueError(
                        "TrajectoryMap.from_dsl: linear b size mismatch. "
                        f"Expected {rows}, got {b_vec.size}."
                    )
            return cls(A=A_mat, b=b_vec, steps=steps, q_dim=q_dim)

        raise ValueError(
            f"TrajectoryMap.from_dsl: unsupported trajectory type {typ!r}. "
            "Supported types: 'bspline', 'linear'."
        )

    @staticmethod
    def _default_clamped_uniform_knots(*, num_ctrl_points: int, degree: int) -> Array:
        knot_count = int(num_ctrl_points + degree + 1)
        knots = np.zeros((knot_count,), dtype=float)
        knots[-(degree + 1) :] = 1.0

        interior = int(num_ctrl_points - degree - 1)
        if interior > 0:
            knots[degree + 1 : degree + 1 + interior] = np.linspace(
                0.0,
                1.0,
                interior + 2,
                dtype=float,
            )[1:-1]
        return knots

    @classmethod
    def _bspline_basis_matrix(
        cls,
        *,
        u_vec: Array,
        degree: int,
        knots: Array,
        num_ctrl_points: int,
    ) -> Array:
        basis = np.zeros((u_vec.size, num_ctrl_points), dtype=float)
        for r, u in enumerate(u_vec):
            basis[r, :] = cls._bspline_basis_row(
                u=float(u),
                degree=degree,
                knots=knots,
                num_ctrl_points=num_ctrl_points,
            )

        basis[np.abs(basis) < 1e-14] = 0.0
        row_sums = np.sum(basis, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-9, rtol=1e-9):
            raise ValueError("TrajectoryMap.from_bspline: invalid basis; rows must sum to 1.")
        return basis

    @staticmethod
    def _bspline_basis_row(
        *,
        u: float,
        degree: int,
        knots: Array,
        num_ctrl_points: int,
    ) -> Array:
        # Cox-de Boor recursion over all control indices.
        N = np.zeros((num_ctrl_points, degree + 1), dtype=float)
        for i in range(num_ctrl_points):
            left = float(knots[i])
            right = float(knots[i + 1])
            in_span = (left <= u < right) or (u == float(knots[-1]) and i == (num_ctrl_points - 1))
            if in_span:
                N[i, 0] = 1.0

        for p in range(1, degree + 1):
            for i in range(num_ctrl_points):
                left = 0.0
                left_den = float(knots[i + p] - knots[i])
                if left_den > 0.0:
                    left = (u - float(knots[i])) / left_den * N[i, p - 1]

                right = 0.0
                if i + 1 < num_ctrl_points:
                    right_den = float(knots[i + p + 1] - knots[i + 1])
                    if right_den > 0.0:
                        right = (float(knots[i + p + 1]) - u) / right_den * N[i + 1, p - 1]

                N[i, p] = left + right

        return N[:, degree]

    @staticmethod
    def _pick_dsl_value(dsl: Mapping[str, Any], *, section: str, key: str) -> Any:
        if key in dsl:
            return dsl[key]
        section_obj = dsl.get(section, None)
        if isinstance(section_obj, Mapping) and key in section_obj:
            return section_obj[key]
        return None

    @staticmethod
    def _resolve_optional_positive_int(value: Any, *, name: str, fallback: int | None = None) -> int | None:
        v = fallback if value is None else value
        if v is None:
            return None
        try:
            out = int(v)
        except Exception as e:
            raise ValueError(f"TrajectoryMap.from_dsl: {name} must be an integer, got {v!r}.") from e
        if out <= 0:
            raise ValueError(f"TrajectoryMap.from_dsl: {name} must be > 0, got {out}.")
        return out

    @staticmethod
    def _resolve_required_positive_int(value: Any, *, name: str) -> int:
        if value is None:
            raise ValueError(f"TrajectoryMap.from_dsl: {name} is required.")
        try:
            out = int(value)
        except Exception as e:
            raise ValueError(f"TrajectoryMap.from_dsl: {name} must be an integer, got {value!r}.") from e
        if out <= 0:
            raise ValueError(f"TrajectoryMap.from_dsl: {name} must be > 0, got {out}.")
        return out

    @staticmethod
    def _resolve_required_nonnegative_int(value: Any, *, name: str) -> int:
        if value is None:
            raise ValueError(f"TrajectoryMap.from_dsl: {name} is required.")
        try:
            out = int(value)
        except Exception as e:
            raise ValueError(f"TrajectoryMap.from_dsl: {name} must be an integer, got {value!r}.") from e
        if out < 0:
            raise ValueError(f"TrajectoryMap.from_dsl: {name} must be >= 0, got {out}.")
        return out
