from __future__ import annotations

import numpy as np

Array = np.ndarray

def scale_matrix_with_projection_svd(
    jac: Array,
    *,
    svd_rtol: float | None = None,
) -> tuple[Array, int]:
    jac_mat = np.asarray(jac, dtype=float)
    if jac_mat.ndim != 2:
        raise ValueError(
            "scale_matrix_with_projection_svd: jac must be 2D, "
            f"got shape {jac_mat.shape}."
        )

    if jac_mat.size == 0:
        return np.zeros((jac_mat.shape[0], 0), dtype=float), 0

    rcond = None if svd_rtol is None else float(svd_rtol)
    proj = jac_mat @ np.linalg.pinv(jac_mat, rcond=rcond)
    u, s, _ = np.linalg.svd(proj, full_matrices=True)

    if s.size == 0:
        rank = 0
    elif svd_rtol is None:
        rank = int(np.linalg.matrix_rank(proj))
    else:
        threshold = float(svd_rtol) * max(proj.shape) * float(s[0])
        rank = int(np.sum(s > threshold))

    jac_scaled = (u * np.sqrt(s))[:, :rank]

    return np.asarray(jac_scaled, dtype=float), rank
