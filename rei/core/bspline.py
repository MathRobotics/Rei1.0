from __future__ import annotations

import numpy as np
from functools import lru_cache

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

    # Cox-de Boor recurrence over all samples and control indices at once.
    # Only the degree loop remains in Python. Preserve the scalar evaluator's
    # right-endpoint convention, including repeated knots.
    u = u_vec[:, None]
    basis = ((knots[:num_ctrl_points] <= u)
             & (u < knots[1:num_ctrl_points + 1])).astype(float)
    basis[u_vec == knots[-1], -1] = 1.
    for p in range(1, degree + 1):
        left_den = knots[p:p + num_ctrl_points] - knots[:num_ctrl_points]
        left = np.divide(u - knots[:num_ctrl_points], left_den,
                         out=np.zeros_like(basis), where=left_den > 0)
        next_basis = left * basis
        right_den = knots[p + 1:p + num_ctrl_points] - knots[1:num_ctrl_points]
        right = np.divide(knots[p + 1:p + num_ctrl_points] - u, right_den,
                          out=np.zeros_like(basis[:, :-1]), where=right_den > 0)
        next_basis[:, :-1] += right * basis[:, 1:]
        basis = next_basis

    basis[np.abs(basis) < 1e-14] = 0.0
    row_sums = np.sum(basis, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-9, rtol=1e-9):
        raise ValueError("bspline_basis_matrix: invalid basis; rows must sum to 1.")
    return basis


def _bspline_derivative_transform(
    *,
    degree: int,
    knots: Array,
    num_ctrl_points: int,
) -> Array:
    """Linear map from control points to derivative control points.

    For degree ``p`` spline control points ``P_i``, derivative control points are:

      D_i = p / (U_{i+p+1} - U_{i+1}) * (P_{i+1} - P_i)
    """

    degree = int(degree)
    num_ctrl_points = int(num_ctrl_points)
    knots = np.asarray(knots, dtype=float).reshape(-1)

    if degree <= 0 or num_ctrl_points <= 1:
        return np.zeros((max(num_ctrl_points - 1, 0), num_ctrl_points), dtype=float)

    out = np.zeros((num_ctrl_points - 1, num_ctrl_points), dtype=float)
    indices = np.arange(num_ctrl_points - 1)
    denom = knots[indices + degree + 1] - knots[indices + 1]
    c = np.divide(float(degree), denom, out=np.zeros_like(denom), where=denom > 0)
    out[indices, indices] = -c
    out[indices, indices + 1] = c
    return out


def bspline_basis_derivative_matrix(
    *,
    u_vec: Array,
    degree: int,
    knots: Array,
    num_ctrl_points: int,
    derivative_order: int = 1,
) -> Array:
    """Evaluate the ``derivative_order``-th derivative of B-spline basis."""

    derivative_order = int(derivative_order)
    if derivative_order < 0:
        raise ValueError(
            "bspline_basis_derivative_matrix: derivative_order must be >= 0, "
            f"got {derivative_order}."
        )
    mats = bspline_basis_derivative_matrices(
        u_vec=u_vec,
        degree=degree,
        knots=knots,
        num_ctrl_points=num_ctrl_points,
        max_derivative_order=derivative_order,
    )
    return mats[derivative_order, :, :]


def bspline_basis_derivative_matrices(
    *,
    u_vec: Array,
    degree: int,
    knots: Array,
    num_ctrl_points: int,
    max_derivative_order: int,
) -> Array:
    """Evaluate derivatives of B-spline basis wrt parameter ``u``.

    Returns tensor ``B_all`` with shape ``(max_derivative_order + 1, len(u_vec), num_ctrl_points)`` where:

      B_all[r, k, i] = d^r N_i,p(u_k) / du^r

    for ``r = 0..max_derivative_order``.
    """

    u_vec = np.asarray(u_vec, dtype=float).reshape(-1)
    degree = int(degree)
    num_ctrl_points = int(num_ctrl_points)
    knots = np.asarray(knots, dtype=float).reshape(-1)
    max_derivative_order = int(max_derivative_order)

    if max_derivative_order < 0:
        raise ValueError(
            "bspline_basis_derivative_matrices: max_derivative_order must be >= 0, "
            f"got {max_derivative_order}."
        )

    if max_derivative_order > degree:
        raise ValueError(
            "B-spline: requested derivative order must be <= degree; "
            f"got derivative order {max_derivative_order} > degree {degree}."
        )

    matrices = bspline_basis_derivative_matrices_for_orders(
        u_vec=u_vec, degree=degree, knots=knots, num_ctrl_points=num_ctrl_points,
        orders=range(max_derivative_order + 1),
    )
    return np.stack([matrices[r] for r in range(max_derivative_order + 1)])


def bspline_basis_derivative_matrices_for_orders(
    *, u_vec: Array, degree: int, knots: Array, num_ctrl_points: int, orders,
) -> dict[int, Array]:
    """Evaluate only requested orders, reusing cached matrices across calls."""
    orders = set(orders)
    if any(not isinstance(r, (int, np.integer)) or r < 0 or r > degree for r in orders):
        raise ValueError(f"B-spline derivative orders must be integers in [0, {degree}].")
    if not orders:
        return {}
    u_vec = np.asarray(u_vec, dtype=float).reshape(-1)
    knots = np.asarray(knots, dtype=float).reshape(-1)
    # Cache immutable basis data only; callers receive independent writable
    # arrays. Exact keys deliberately avoid merging nearby knot/sample grids.
    return {order: _cached_derivative_basis(
        int(degree), int(num_ctrl_points), knots.tobytes(), u_vec.tobytes(), int(order)
    ).copy() for order in orders}


@lru_cache(maxsize=64)
def _cached_derivative_basis(degree, num_ctrl_points, knot_bytes, sample_bytes, order):
    result = _derivative_matrices_uncached(
        u_vec=np.frombuffer(sample_bytes, dtype=float), degree=degree,
        knots=np.frombuffer(knot_bytes, dtype=float), num_ctrl_points=num_ctrl_points,
        orders={order},
    )[order]
    result.setflags(write=False)
    return result


def _derivative_matrices_uncached(*, u_vec, degree, knots, num_ctrl_points, orders):
    out = {}
    if 0 in orders:
        out[0] = bspline_basis_matrix(u_vec=u_vec, degree=degree, knots=knots,
                                      num_ctrl_points=num_ctrl_points)

    current_knots = knots.copy()
    current_degree = degree
    current_num_ctrl = num_ctrl_points
    transform = np.eye(num_ctrl_points, dtype=float)
    for order in range(1, max(orders) + 1):
        D = _bspline_derivative_transform(
            degree=current_degree,
            knots=current_knots,
            num_ctrl_points=current_num_ctrl,
        )
        transform = D @ transform
        current_knots = current_knots[1:-1]
        current_degree -= 1
        current_num_ctrl -= 1

        if current_num_ctrl <= 0:
            break

        if order not in orders:
            continue

        basis_low = bspline_basis_matrix(
            u_vec=u_vec,
            degree=current_degree,
            knots=current_knots,
            num_ctrl_points=current_num_ctrl,
        )
        out[order] = basis_low @ transform

    for matrix in out.values():
        matrix[np.abs(matrix) < 1e-14] = 0.0
    return out
