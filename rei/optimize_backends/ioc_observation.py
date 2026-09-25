"""Reproducible observation noise between forward optimization and IOC."""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..core.trajectory import BsplineTrajectoryOperator, TrajectoryMap


@dataclass
class NoisyIocTrajectory:
    """Sampled joint observations and their least-squares trajectory fit.

    Arrays are independent of the input. ``p`` contains trajectory parameters,
    not a full variable pack or nullspace-reduced solver coordinates.
    """

    p: np.ndarray
    q_clean: np.ndarray
    q_observed: np.ndarray
    q_fitted: np.ndarray
    noise: np.ndarray
    std: np.ndarray
    seed: int | None
    fit_rank: int

    @property
    def fit_residual_norm(self) -> float:
        return float(np.linalg.norm(self.q_fitted - self.q_observed))


def prepare_noisy_ioc_trajectory(
    trajectory_map: TrajectoryMap,
    p: np.ndarray,
    *,
    std: float | np.ndarray,
    seed: int | None = None,
) -> NoisyIocTrajectory:
    """Add independent zero-mean Gaussian noise to q(k), then refit the map.

    ``std`` is a nonnegative scalar or a vector of length ``q_dim``, in joint
    coordinate units (radians for revolute joints, metres for prismatic joints).
    Every sample, including the endpoints, receives noise. The unconstrained
    least-squares fit does not enforce DOC boundary conditions or joint limits.
    Velocity/acceleration for IOC are derived from the fitted parameters using
    the existing derivative maps, not independently perturbed observations.

    The fit solves A delta_p ~= noise and returns p + delta_p. This preserves
    the original nullspace component when A is rank deficient, and preserves p
    exactly for zero noise. B-spline maps solve using the scalar basis without
    constructing a Kronecker matrix. Inputs and global RNG state are unchanged.
    """
    if not isinstance(trajectory_map, TrajectoryMap):
        raise TypeError("trajectory_map must be a TrajectoryMap.")
    point = np.asarray(p, dtype=float)
    if point.shape != (trajectory_map.p_dim,) or not np.all(np.isfinite(point)):
        raise ValueError("p must be a finite vector of length trajectory_map.p_dim.")
    sigma = np.asarray(std, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(trajectory_map.q_dim, float(sigma))
    if sigma.shape != (trajectory_map.q_dim,) or not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError("std must be finite and nonnegative: a scalar or a vector of length q_dim.")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None.")
    seed = None if seed is None else int(seed)
    shape = (trajectory_map.steps, trajectory_map.q_dim)
    clean = trajectory_map.apply(point).reshape(shape)
    operator = trajectory_map.A
    matrix = operator.basis if isinstance(operator, BsplineTrajectoryOperator) else operator
    if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(matrix)):
        raise ValueError("trajectory_map must contain only finite values.")
    noise = np.random.Generator(np.random.PCG64(seed)).normal(size=shape) * sigma
    observed = clean + noise
    rhs = noise if isinstance(operator, BsplineTrajectoryOperator) else noise.reshape(-1)
    delta, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    if isinstance(operator, BsplineTrajectoryOperator):
        rank *= trajectory_map.q_dim
    fitted_p = point + delta.reshape(-1)
    fitted_q = trajectory_map.apply(fitted_p).reshape(shape)
    if not all(np.all(np.isfinite(v)) for v in (observed, fitted_p, fitted_q)):
        raise ValueError("Noise or fitted trajectory is non-finite; reduce std or check map conditioning.")
    return NoisyIocTrajectory(
        p=fitted_p, q_clean=clean, q_observed=observed, q_fitted=fitted_q,
        noise=noise, std=sigma.copy(), seed=seed, fit_rank=int(rank),
    )
