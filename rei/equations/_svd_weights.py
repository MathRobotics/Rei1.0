"""SVD fitting of a single nonnegative stationarity-weight direction."""
from __future__ import annotations

from typing import Any

import numpy as np


class SVDWeightError(ValueError):
    """An SVD direction cannot be used as weights; diagnostics remain available."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def solve_svd_weights(
    matrix: np.ndarray, *, rtol: float | None, nonnegative_tol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2 or not np.all(np.isfinite(A)):
        raise ValueError("SVD weight fitting requires a finite 2D gradient matrix.")
    if rtol is not None and (not np.isfinite(rtol) or not 0 <= rtol < 1):
        raise ValueError("svd_rtol must be None or finite in [0, 1).")
    if not np.isfinite(nonnegative_tol) or not 0 <= nonnegative_tol < 1:
        raise ValueError("svd_nonnegative_tol must be finite in [0, 1).")
    rows, cols = A.shape
    diagnostics: dict[str, Any] = {"matrix_shape": [rows, cols], "status": "no_active_terms",
                                   "singular_values": []}
    if not cols:
        raise SVDWeightError("SVD weight fitting requires at least one active objective term.", diagnostics)
    # Tall matrices need only the reduced U. Wide matrices need the full Vh
    # to retain right-nullspace vectors absent from the reduced decomposition.
    _, singular_values, vh = np.linalg.svd(A, full_matrices=rows < cols)
    if not np.all(np.isfinite(singular_values)):
        raise FloatingPointError("SVD returned non-finite singular values.")
    relative_tol = np.finfo(float).eps * max(A.shape) if rtol is None else float(rtol)
    threshold = relative_tol * float(singular_values[0]) if singular_values.size else 0.0
    right_values = np.pad(singular_values, (0, cols - singular_values.size))
    smallest = float(right_values[-1])
    multiplicity = int(np.count_nonzero(right_values - smallest <= threshold))
    rank = int(np.count_nonzero(singular_values > threshold))
    diagnostics.update(
        singular_values=singular_values.tolist(), implicit_zero_singular_values=cols - singular_values.size,
        rtol=relative_tol, rank_tolerance=threshold, rank=rank, nullity=cols - rank,
        smallest_singular_value=smallest, smallest_multiplicity=multiplicity,
        unique_direction=multiplicity == 1, nonnegative_tolerance=float(nonnegative_tol),
    )
    if multiplicity != 1:
        diagnostics["status"] = "non_unique_direction"
        raise SVDWeightError(
            f"SVD smallest singular value has multiplicity {multiplicity}; "
            "a unique weight direction cannot be selected from one singular vector.", diagnostics,
        )
    weights = vh[-1].copy()
    if weights.sum() < 0:
        weights = -weights
    diagnostics["signed_direction"] = weights.tolist()
    if np.any(weights < -nonnegative_tol):
        diagnostics["status"] = "mixed_signs"
        raise SVDWeightError("SVD weight direction has mixed signs; nonnegative weights cannot be adopted.", diagnostics)
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        diagnostics["status"] = "invalid_normalization"
        raise SVDWeightError("SVD weight direction cannot be normalized to unit sum.", diagnostics)
    diagnostics["status"] = "estimated"
    return weights / total, diagnostics
