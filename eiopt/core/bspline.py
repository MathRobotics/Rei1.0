from __future__ import annotations

import numpy as np

Array = np.ndarray


def default_clamped_uniform_knots(*, num_ctrl_points: int, degree: int) -> Array:
    """Create a clamped uniform knot vector in [0, 1]."""

    num_ctrl_points = int(num_ctrl_points)
    degree = int(degree)
    if num_ctrl_points <= 0:
        raise ValueError(f"default_clamped_uniform_knots: num_ctrl_points must be > 0, got {num_ctrl_points}.")
    if degree < 0:
        raise ValueError(f"default_clamped_uniform_knots: degree must be >= 0, got {degree}.")
    if num_ctrl_points < degree + 1:
        raise ValueError(
            "default_clamped_uniform_knots: num_ctrl_points must satisfy "
            f"num_ctrl_points >= degree + 1, got {num_ctrl_points} and {degree}."
        )

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


def bspline_basis_row(
    *,
    u: float,
    degree: int,
    knots: Array,
    num_ctrl_points: int,
) -> Array:
    """Evaluate all B-spline basis functions N_i,p(u)."""

    degree = int(degree)
    num_ctrl_points = int(num_ctrl_points)
    knots = np.asarray(knots, dtype=float).reshape(-1)

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


def bspline_basis_matrix(
    *,
    u_vec: Array,
    degree: int,
    knots: Array,
    num_ctrl_points: int,
) -> Array:
    """Evaluate the basis matrix B where B[k, i] = N_i,p(u_k)."""

    u_vec = np.asarray(u_vec, dtype=float).reshape(-1)
    degree = int(degree)
    num_ctrl_points = int(num_ctrl_points)
    knots = np.asarray(knots, dtype=float).reshape(-1)

    basis = np.zeros((u_vec.size, num_ctrl_points), dtype=float)
    for r, u in enumerate(u_vec):
        basis[r, :] = bspline_basis_row(
            u=float(u),
            degree=degree,
            knots=knots,
            num_ctrl_points=num_ctrl_points,
        )

    basis[np.abs(basis) < 1e-14] = 0.0
    row_sums = np.sum(basis, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-9, rtol=1e-9):
        raise ValueError("bspline_basis_matrix: invalid basis; rows must sum to 1.")
    return basis
