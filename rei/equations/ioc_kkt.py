"""Inverse KKT fitting with simplex weights and signed constraint multipliers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ioc import _contribution_json, _scale_columns
from .stationarity import (
    StationaritySource,
    build_stationarity_gradient_matrix,
    filter_stationarity_contributions,
    term_constraint_kind,
)


@dataclass(frozen=True)
class KktConstraintData:
    """Unweighted constraints h=0, g<=0, with Jacobians shaped (rows, variables).

    Use signed inequality margins, never clipped penalty residuals. Row order
    determines the order of the reported multipliers.
    """

    equality_values: Any
    equality_jacobian: Any
    inequality_values: Any
    inequality_jacobian: Any


def _validated_constraints(data: KktConstraintData, n: int):
    arrays = []
    for kind, values, jacobian in (
        ("equality", data.equality_values, data.equality_jacobian),
        ("inequality", data.inequality_values, data.inequality_jacobian),
    ):
        v = np.asarray(values, dtype=float)
        J = np.asarray(jacobian, dtype=float)
        if v.ndim != 1 or J.shape != (v.size, n):
            raise ValueError(f"{kind} values/Jacobian must have shapes (rows,) and (rows, {n}).")
        if not np.all(np.isfinite(v)) or not np.all(np.isfinite(J)):
            raise ValueError(f"{kind} values/Jacobian must be finite.")
        arrays.extend((v, J))
    return arrays


def _column_basis(matrix):
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    threshold = np.finfo(float).eps * max(matrix.shape, default=0) * np.max(s, initial=0.0)
    return u[:, s > threshold]


def _fit_nonnegative_affine(B, n_weights, *, max_iters, tol):
    """Minimize ||Bz|| with z>=0 and sum(z[:n_weights])=1.

    A feasible active-set method solves each face by nullspace least squares;
    no normal equations or external optimizer are needed.
    """
    c = np.r_[np.ones(n_weights), np.zeros(B.shape[1] - n_weights)]
    z = c / n_weights
    free = c.astype(bool)
    for iteration in range(1, max_iters + 1):
        indices = np.flatnonzero(free)
        cf = c[indices]
        particular = cf / (cf @ cf)
        # The last rows of Vh span the nullspace of the simplex equality.
        _, _, vh = np.linalg.svd(cf[None, :], full_matrices=True)
        Z = vh[1:].T
        Bf = B[:, indices]
        candidate = np.zeros_like(z)
        candidate[indices] = particular + Z @ np.linalg.lstsq(
            Bf @ Z, -(Bf @ particular), rcond=None,
        )[0]
        negative = indices[candidate[indices] < -tol]
        if negative.size:
            delta = candidate - z
            alpha = np.min(z[negative] / -delta[negative])
            z += alpha * delta
            blocked = free & (z <= tol) & (delta < 0)
            free[blocked] = False
            z[blocked] = 0.0
            continue
        z = np.maximum(candidate, 0.0)
        z /= c @ z
        gradient = B.T @ (B @ z)
        multiplier = -(cf @ gradient[indices]) / (cf @ cf)
        reduced = gradient + multiplier * c
        inactive = np.flatnonzero(~free)
        if not inactive.size or np.min(reduced[inactive]) >= -tol:
            return z, iteration, True
        free[inactive[np.argmin(reduced[inactive])]] = True
    return z, max_iters, False


def estimate_kkt_weights(
    source: StationaritySource,
    constraints: KktConstraintData,
    *,
    p: Any | None = None,
    stationarity_scaling: str = "gradient_norm",
    stationarity_scale_eps: float = 1e-12,
    active_tol: float = 1e-8,
    feasibility_tol: float = 1e-8,
    max_iters: int = 2000,
    tol: float = 1e-10,
) -> dict[str, Any]:
    """Fit A w + Jh.T lambda + Jg.T mu = 0 at one demonstration.

    Weights are nonnegative and sum to one; equality multipliers are free,
    inequality multipliers are nonnegative and zero on inactive rows. All
    objective columns, including zero gradients, participate. Constraint data
    must describe the point supplied to the source. Multipliers use the same
    half-squared-residual convention as StationarityTermContribution (J.T r).

    Exact KKT satisfaction is necessary, not sufficient, for a minimum. The
    rank-based uniqueness check is conservative at nonnegative boundaries.
    """
    for name, value in (("active_tol", active_tol), ("feasibility_tol", feasibility_tol), ("tol", tol)):
        if not np.isfinite(value) or value < 0 or (name == "tol" and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if name == 'tol' else 'nonnegative'}.")
    if isinstance(max_iters, bool) or not np.isfinite(max_iters) or int(max_iters) != max_iters or max_iters < 1:
        raise ValueError("max_iters must be a positive integer.")
    if p is not None:
        source.set_point(np.asarray(p, dtype=float).reshape(-1))
    all_terms = source.term_contributions(required=source.required_list(None))
    terms = filter_stationarity_contributions(all_terms, include_constraints=False)
    A, term_indices = build_stationarity_gradient_matrix(terms, n_total=source.n_total)
    if not terms:
        raise ValueError("KKT IOC requires at least one objective term.")
    if not np.all(np.isfinite(A)):
        raise ValueError("Objective gradients must be finite.")
    h, Jh, g, Jg = _validated_constraints(constraints, source.n_total)
    active = np.flatnonzero(g >= -active_tol)
    As, scales = _scale_columns(A, terms, mode=stationarity_scaling, eps=stationarity_scale_eps)
    normals = Jg[active].T
    # Scaling multiplier columns improves conditioning without changing signs.
    mu_scales = 1.0 / np.maximum(np.linalg.norm(normals, axis=0), stationarity_scale_eps)
    C = np.column_stack((As, normals * mu_scales))
    equality_basis = _column_basis(Jh.T)
    B = C - equality_basis @ (equality_basis.T @ C)
    # Remove cancellation at machine precision after projection.
    B[np.abs(B) <= np.finfo(float).eps * max(C.shape) * np.max(np.abs(C), initial=0.0)] = 0.0
    coefficients, iterations, converged = _fit_nonnegative_affine(
        B, len(terms), max_iters=int(max_iters), tol=tol,
    )
    scaled_weights = coefficients[:len(terms)]
    normalizer = float(scales @ scaled_weights)
    weights = scales * scaled_weights / normalizer
    mu = np.zeros_like(g)
    mu[active] = mu_scales * coefficients[len(terms):] / normalizer
    objective_gradient = A @ weights
    lam = np.linalg.lstsq(Jh.T, -(objective_gradient + Jg.T @ mu), rcond=None)[0]
    residual = objective_gradient + Jh.T @ lam + Jg.T @ mu
    scaled_residual = residual * normalizer
    stationarity_inf = float(np.max(np.abs(residual), initial=0.0))
    eq_violation = float(np.max(np.abs(h), initial=0.0))
    ineq_violation = float(np.max(g, initial=0.0))
    complementarity = float(np.max(np.abs(mu * g), initial=0.0))
    ok = converged and stationarity_inf <= tol and max(eq_violation, ineq_violation, complementarity) <= feasibility_tol

    augmented = np.vstack((B, np.r_[np.ones(len(terms)), np.zeros(active.size)]))
    rank = int(np.linalg.matrix_rank(augmented))
    mu_rank = int(np.linalg.matrix_rank(augmented[:, len(terms):])) if active.size else 0
    weight_nullity = len(terms) - (rank - mu_rank)
    messages = []
    if not converged:
        messages.append("KKT weight fitting reached the iteration limit.")
    if not ok:
        messages.append("The fitted weights and multipliers do not satisfy all KKT checks at this point.")
    if weight_nullity:
        messages.append("Weights are not uniquely determined by the linear KKT equations.")
    return {
        "terms": [_contribution_json(t, weight=float(w)) for t, w in zip(terms, weights)],
        "active_terms": [_contribution_json(t, weight=float(w)) for t, w in zip(terms, weights)],
        "weights": weights.tolist(),
        "scaled_weights": scaled_weights.tolist(),
        "stationarity_scaling": {
            "mode": stationarity_scaling, "eps": float(stationarity_scale_eps),
            "column_scale": scales.tolist(), "active_column_scale": scales.tolist(),
        },
        "stationarity": {
            "term_indices": list(term_indices), "active_indices": list(range(len(terms))),
            "active_gradient_indices": [i for i, t in enumerate(terms) if np.linalg.norm(t.gradient) > tol],
            "active_residual_indices": [i for i, t in enumerate(terms) if (t.weighted_residual_norm or 0) > tol],
            "objective_gradient": objective_gradient.tolist(),
            "ikkt_residual": residual.tolist(), "ikkt_residual_scaled": scaled_residual.tolist(),
            "ikkt_residual_norm": float(np.linalg.norm(residual)),
            "ikkt_residual_scaled_norm": float(np.linalg.norm(scaled_residual)),
        },
        "simplex": None,
        "kkt": {
            "ok": bool(ok), "status": "converged" if converged else "max_iters",
            "iterations": iterations, "active_inequality_rows": active.tolist(),
            "equality_values": h.tolist(), "inequality_values": g.tolist(),
            "equality_multipliers": lam.tolist(), "inequality_multipliers": mu.tolist(),
            "stationarity_inf": stationarity_inf, "eq_violation_inf": eq_violation,
            "ineq_violation_inf": ineq_violation, "complementarity_inf": complementarity,
            "active_tol": float(active_tol), "feasibility_tol": float(feasibility_tol),
            "stationarity_tol": float(tol),
        },
        "validation": {
            "scope": "constrained_kkt", "constraint_kkt_checked": True,
            "constraint_term_indices": [t.term_index for t in all_terms if term_constraint_kind(t.attrs)[0]],
            "messages": messages,
        },
        "identifiability": {
            "active_augmented_rank": rank, "active_affine_nullity": weight_nullity,
            "weight_affine_nullity": weight_nullity, "excluded_term_indices": [],
            "unique_stationary_weights": bool(ok and weight_nullity == 0),
        },
    }


__all__ = ["KktConstraintData", "estimate_kkt_weights"]
