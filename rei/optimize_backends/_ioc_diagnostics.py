"""Diagnostics for an IOC fit, independent of backend compilation/solving."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..equations.stationarity import StationarityTermContribution, term_constraint_kind


def diagnose_ioc_fit(
    *,
    contributions_all: list[StationarityTermContribution],
    contributions: list[StationarityTermContribution],
    active_idx: tuple[int, ...],
    A_scaled: np.ndarray,
    ikkt_residual_scaled: np.ndarray,
    simplex_tol: float,
) -> dict[str, Any]:
    constraint_indices = [
        int(term.term_index) for term in contributions_all
        if term_constraint_kind(term.attrs)[0]
    ]
    active_matrix = A_scaled[:, list(active_idx)]
    # Rank of [A; 1^T] checks uniqueness under the sum-to-one condition.
    # Deficiency alone does not prove multiple nonnegative solutions at a boundary.
    if active_idx:
        magnitude = max(float(np.max(np.abs(active_matrix), initial=0.0)), 1.0)
        augmented = np.vstack((active_matrix / magnitude, np.ones((1, len(active_idx)))))
        singular_values = np.linalg.svd(augmented, compute_uv=False)
        threshold = np.finfo(float).eps * max(augmented.shape) * singular_values[0]
        rank = int(np.count_nonzero(singular_values > threshold))
    else:
        singular_values = np.zeros(0)
        rank = 0
    affine_nullity = len(active_idx) - rank
    messages = []
    if constraint_indices:
        messages.append("Constraint KKT multipliers are not fitted; objective stationarity alone is insufficient.")
    if affine_nullity:
        messages.append("Active weights are not uniquely determined by the linear stationarity equations.")
    if len(active_idx) != len(contributions):
        messages.append("Inactive terms were excluded; their weights are not identified by this fit.")
    if np.linalg.norm(ikkt_residual_scaled) > simplex_tol:
        messages.append("The fitted coefficients do not satisfy exact scaled stationarity.")
    if not active_idx:
        messages.append("No active terms; the returned zero weights are not a simplex estimate.")

    return {
        "validation": {
            "scope": "objective_stationarity",
            "constraint_term_indices": constraint_indices,
            "constraint_kkt_checked": False,
            "messages": messages,
        },
        "identifiability": {
            "active_augmented_rank": rank,
            "active_affine_nullity": affine_nullity,
            "active_augmented_singular_values": singular_values.tolist(),
            "excluded_term_indices": [
                int(term.term_index) for i, term in enumerate(contributions) if i not in active_idx
            ],
            "unique_stationary_weights": bool(active_idx)
            and not constraint_indices
            and affine_nullity == 0
            and len(active_idx) == len(contributions)
            and float(np.linalg.norm(ikkt_residual_scaled)) <= simplex_tol,
        },
    }
