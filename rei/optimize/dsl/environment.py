from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...core.time_grid import TimeGrid
from ...core.trajectory import TrajectoryMap
from ...core.expr.registry import ExprRegister
from ...core.expr.types import VariablePack
from .trajectory import (
    build_trajectory_map,
    build_trajectory_maps_with_derivatives,
    default_dt_from_time,
    default_steps_from_time,
)


TrajectorySpecFingerprint = bytes
TrajectoryCacheKey = tuple[int, int | None, int | None]
TrajectoryDerivativeCacheKey = tuple[
    TrajectorySpecFingerprint,
    int,
    str,
    float | None,
    int | None,
    int | None,
]


def _trajectory_spec_fingerprint(value: Any) -> TrajectorySpecFingerprint:
    """Return a content fingerprint suitable for compile-local trajectory caches."""

    digest = hashlib.blake2b(digest_size=16)

    def update(item: Any) -> None:
        if isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
            digest.update(b"}")
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray:")
            digest.update(array.dtype.str.encode("utf-8"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(memoryview(array).cast("B"))
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"list[" if isinstance(item, list) else b"tuple(")
            for child in item:
                update(child)
            digest.update(b"]" if isinstance(item, list) else b")")
            return
        if isinstance(item, np.generic):
            update(item.item())
            return
        digest.update(type(item).__module__.encode("utf-8"))
        digest.update(b".")
        digest.update(type(item).__qualname__.encode("utf-8"))
        digest.update(b":")
        digest.update(repr(item).encode("utf-8"))
        digest.update(b";")

    update(value)
    return digest.digest()


def _normalize_derivative_wrt(value: str) -> str:
    wrt = str(value).strip().lower()
    if wrt in ("u", "param", "parameter"):
        return "u"
    return wrt


@dataclass
class DslBuildEnv:
    pack: VariablePack
    time: TimeGrid
    expr_register: ExprRegister
    root_dsl: dict[str, Any] | None = None
    trajectory_cache: dict[TrajectoryCacheKey, TrajectoryMap] = field(default_factory=dict)
    trajectory_derivative_cache: dict[TrajectoryDerivativeCacheKey, TrajectoryMap] = field(default_factory=dict)

    def build_expr(self, dsl: dict[str, Any]) -> Any:
        typ = str(dsl["type"])
        builder = self.expr_register.expr.get(typ, None)
        if builder is None:
            raise ValueError(f"unknown expr type: {typ}")
        return builder(self, dsl)

    def build_cost(self, dsl: dict[str, Any]) -> Any:
        typ = str(dsl.get("type", "l2"))
        builder = self.expr_register.cost.get(typ, None)
        if builder is None:
            raise ValueError(f"unknown cost type: {typ}")
        return builder(dsl)

    def resolve_trajectory_map(
        self,
        traj_dsl: Mapping[str, Any],
        *,
        default_q_dim: int | None = None,
    ) -> TrajectoryMap:
        default_steps = default_steps_from_time(self.time)
        key = (id(traj_dsl), default_steps, default_q_dim)
        cached = self.trajectory_cache.get(key, None)
        if cached is not None:
            return cached

        traj = build_trajectory_map(
            traj_dsl,
            default_steps=default_steps,
            default_q_dim=default_q_dim,
        )
        self.trajectory_cache[key] = traj
        return traj

    def resolve_trajectory_map_with_derivative(
        self,
        traj_dsl: Mapping[str, Any],
        *,
        derivative_order: int,
        derivative_wrt: str = "u",
        default_q_dim: int | None = None,
    ) -> TrajectoryMap:
        deriv_order = int(derivative_order)
        if deriv_order < 0:
            raise ValueError(f"derivative_order must be >= 0, got {deriv_order}.")
        if deriv_order == 0:
            return self.resolve_trajectory_map(traj_dsl, default_q_dim=default_q_dim)

        default_steps = default_steps_from_time(self.time)
        default_dt = default_dt_from_time(self.time)
        fingerprint = _trajectory_spec_fingerprint(traj_dsl)
        wrt = _normalize_derivative_wrt(derivative_wrt)
        dt_key = default_dt if wrt == "time" else None
        key = (
            fingerprint,
            deriv_order,
            wrt,
            dt_key,
            default_steps,
            default_q_dim,
        )
        cached = self.trajectory_derivative_cache.get(key, None)
        if cached is not None:
            return cached

        trajectories = self.resolve_trajectory_maps_with_derivatives(
            traj_dsl,
            max_derivative_order=deriv_order,
            derivative_wrt=wrt,
            default_q_dim=default_q_dim,
        )
        return trajectories[deriv_order]

    def resolve_trajectory_maps_with_derivatives(
        self,
        traj_dsl: Mapping[str, Any],
        *,
        max_derivative_order: int,
        derivative_wrt: str = "u",
        default_q_dim: int | None = None,
    ) -> list[TrajectoryMap]:
        max_order = int(max_derivative_order)
        if max_order < 0:
            raise ValueError(f"max_derivative_order must be >= 0, got {max_order}.")
        if max_order == 0:
            return [self.resolve_trajectory_map(traj_dsl, default_q_dim=default_q_dim)]

        default_steps = default_steps_from_time(self.time)
        default_dt = default_dt_from_time(self.time)
        fingerprint = _trajectory_spec_fingerprint(traj_dsl)
        wrt = _normalize_derivative_wrt(derivative_wrt)
        dt_key = default_dt if wrt == "time" else None

        keys = [
            (
                fingerprint,
                order,
                wrt,
                dt_key,
                default_steps,
                default_q_dim,
            )
            for order in range(0, max_order + 1)
        ]
        cached_trajectories = [self.trajectory_derivative_cache.get(key, None) for key in keys]
        if all(trajectory is not None for trajectory in cached_trajectories):
            return [trajectory for trajectory in cached_trajectories if trajectory is not None]

        trajectories = build_trajectory_maps_with_derivatives(
            traj_dsl,
            max_derivative_order=max_order,
            derivative_wrt=wrt,
            default_steps=default_steps,
            default_q_dim=default_q_dim,
            default_dt=default_dt,
        )
        base_key = (id(traj_dsl), default_steps, default_q_dim)
        self.trajectory_cache[base_key] = trajectories[0]
        for key, trajectory in zip(keys, trajectories, strict=True):
            self.trajectory_derivative_cache[key] = trajectory
        return list(trajectories)
