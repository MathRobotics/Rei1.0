from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ...core.state_cache import StateKey
from ...core.state_schema import split_jac_field
from ...core.trajectory import TrajectoryMap

Array = np.ndarray


class TrajectoryStateBuilderMixin:
    """Shared build loop for trajectory-parameterized state builders.

    Subclasses are expected to provide:
      - `trajectory_map`
      - `_extract_q`, `_route_for_key`, `_state_ref`, and `_dispatch`
      - `_compose_motion_and_jac(p, k=...)`
      - `_chain_param_jac(...)`

    Backend-specific behavior is kept in hooks so Kots can use matvec fast
    paths, Pinocchio can manage active motion state, and callback providers can
    update custom models.
    """

    trajectory_map: TrajectoryMap

    def _trajectory_error_prefix(self) -> str:
        return type(self).__name__

    def _expected_steps(self, *, time: Any = None) -> int:
        return expected_trajectory_steps(
            self.trajectory_map,
            time=time,
            error_prefix=self._trajectory_error_prefix(),
        )

    def _accept_required_key_for_traj(self, key: StateKey, *, steps: int) -> bool:
        return accept_trajectory_state_key(key, steps=steps)

    def _is_param_jac_key(self, key: StateKey) -> bool:
        return is_param_jacobian_key(key, p_var=getattr(self, "q_var"))

    def _validate_trajectory_parameter_size(self, p: Array) -> None:
        p_vec = np.asarray(p, dtype=float).reshape(-1)
        if p_vec.size != self.trajectory_map.p_dim:
            raise ValueError(
                f"{self._trajectory_error_prefix()}: parameter size mismatch. "
                f"Expected p_dim={self.trajectory_map.p_dim}, got {p_vec.size}."
            )

    def _update_trajectory_step(self, *, k: int, q_k: Array, motion_k: Array) -> None:
        del k, motion_k
        self._update_kinematics(q_k)

    def _finalize_trajectory_build(self) -> None:
        return None

    def _evaluate_trajectory_entry(
        self,
        *,
        key: StateKey,
        entry: Any,
        state_ref: Any,
        q_k: Array,
        dqdp_k: Array,
        dmotiondp_k: Array,
    ) -> Any:
        value = entry.handler(q_k, key, state_ref)
        if self._is_param_jac_key(key):
            value = self._chain_param_jac(
                value,
                key=key,
                jacobian_wrt=entry.jacobian_wrt,
                dqdp_k=dqdp_k,
                dmotiondp_k=dmotiondp_k,
            )
        return value

    def build_state(
        self,
        x_all: Array,
        *,
        pack: Any = None,
        time: Any = None,
        required: Iterable[StateKey] | None = None,
    ) -> dict[StateKey, Any]:
        if required is None:
            return {}

        steps = self._expected_steps(time=time)
        p = self._extract_q(x_all, pack=pack)
        self._validate_trajectory_parameter_size(p)

        grouped: dict[int, list[tuple[StateKey, Any]]] = {}
        for key in required:
            if not self._accept_required_key_for_traj(key, steps=steps):
                continue
            route = self._route_for_key(key)
            if route is None:
                continue
            entry = self._dispatch.get(route, None)
            if entry is None:
                continue
            grouped.setdefault(int(key.k), []).append((key, entry))

        out: dict[StateKey, Any] = {}
        try:
            for k in sorted(grouped.keys()):
                q_k = np.asarray(self.trajectory_map.q_at(p, k), dtype=float).reshape(-1)
                dqdp_k = np.asarray(self.trajectory_map.dqdp_at(k), dtype=float)
                motion_k, dmotiondp_k = self._compose_motion_and_jac(p, k=k)
                self._update_trajectory_step(k=k, q_k=q_k, motion_k=motion_k)

                for key, entry in grouped[k]:
                    state_ref = self._state_ref(key, state_ref_field=entry.state_ref_field)
                    out[key] = self._evaluate_trajectory_entry(
                        key=key,
                        entry=entry,
                        state_ref=state_ref,
                        q_k=q_k,
                        dqdp_k=dqdp_k,
                        dmotiondp_k=dmotiondp_k,
                    )
        finally:
            self._finalize_trajectory_build()

        return out


def validate_trajectory_derivative_maps(
    maps: Mapping[int, TrajectoryMap],
    *,
    error_prefix: str,
) -> None:
    base = maps.get(0, None)
    if base is None:
        raise ValueError(f"{error_prefix}: derivative map for order 0 is required.")
    for order, traj in maps.items():
        if traj.p_dim != base.p_dim:
            raise ValueError(
                f"{error_prefix}: derivative map p_dim mismatch. "
                f"order={order}, expected {base.p_dim}, got {traj.p_dim}."
            )
        if traj.steps != base.steps or traj.q_dim != base.q_dim:
            raise ValueError(
                f"{error_prefix}: derivative map shape mismatch. "
                f"order={order}, expected steps={base.steps}, q_dim={base.q_dim}, "
                f"got steps={traj.steps}, q_dim={traj.q_dim}."
            )


def expected_trajectory_steps(trajectory_map: TrajectoryMap, *, time: Any = None, error_prefix: str) -> int:
    steps = int(trajectory_map.steps)
    if time is None or not hasattr(time, "N"):
        return steps
    try:
        time_steps = int(time.N) + 1
    except Exception:
        return steps
    if time_steps != steps:
        raise ValueError(
            f"{error_prefix}: time grid mismatch. "
            f"trajectory_map.steps={steps}, time steps={time_steps} (N+1)."
        )
    return steps


def accept_trajectory_state_key(key: StateKey, *, steps: int) -> bool:
    if not isinstance(key, StateKey):
        return False
    k = int(getattr(key, "k", -1))
    if k < 0 or k >= int(steps):
        return False
    dtype = getattr(key, "dtype", None)
    if not isinstance(dtype, str) or dtype == "":
        return False
    owner = getattr(key, "owner", None)
    owner_type = getattr(owner, "owner_type", None)
    owner_name = getattr(owner, "owner_name", None)
    if not isinstance(owner_type, str) or owner_type == "":
        return False
    if not isinstance(owner_name, str) or owner_name == "":
        return False
    field = getattr(key, "field", None)
    if not isinstance(field, str) or field == "":
        return False
    return True


def is_param_jacobian_key(key: StateKey, *, p_var: str) -> bool:
    field = getattr(key, "field", None)
    if not isinstance(field, str) or field == "":
        return False
    try:
        _base, var = split_jac_field(field)
    except ValueError:
        return False
    return var == str(p_var)


def compose_interleaved_motion_and_jac(
    p: Array,
    *,
    trajectory_map: TrajectoryMap,
    trajectory_derivative_maps: Mapping[int, TrajectoryMap],
    order: int,
    k: int,
    error_prefix: str,
) -> tuple[Array, Array]:
    p_vec = np.asarray(p, dtype=float).reshape(-1)
    dof = int(trajectory_map.q_dim)
    p_dim = int(trajectory_map.p_dim)
    order_i = int(order)
    motion = np.zeros((dof * order_i,), dtype=float)
    dmotion_dp = np.zeros((dof * order_i, p_dim), dtype=float)

    for deriv_order, traj in trajectory_derivative_maps.items():
        deriv_order_i = int(deriv_order)
        if deriv_order_i >= order_i:
            continue
        q_r = np.asarray(traj.q_at(p_vec, k), dtype=float).reshape(-1)
        J_r = np.asarray(traj.dqdp_at(k), dtype=float)
        if q_r.size != dof:
            raise ValueError(
                f"{error_prefix}: derivative map q size mismatch. "
                f"order={deriv_order_i}, expected {dof}, got {q_r.size}."
            )
        if J_r.shape != (dof, p_dim):
            raise ValueError(
                f"{error_prefix}: derivative map jacobian shape mismatch. "
                f"order={deriv_order_i}, expected {(dof, p_dim)}, got {J_r.shape}."
            )
        motion[deriv_order_i::order_i] = q_r
        dmotion_dp[deriv_order_i::order_i, :] = J_r

    return motion, dmotion_dp


def compose_stacked_motion_and_jac(
    p: Array,
    *,
    trajectory_map: TrajectoryMap,
    trajectory_derivative_maps: Mapping[int, TrajectoryMap],
    derivative_orders: tuple[int, ...],
    k: int,
    error_prefix: str,
) -> tuple[Array, Array]:
    p_vec = np.asarray(p, dtype=float).reshape(-1)
    dof = int(trajectory_map.q_dim)
    p_dim = int(trajectory_map.p_dim)
    motion = np.zeros((len(derivative_orders) * dof,), dtype=float)
    dmotion_dp = np.zeros((len(derivative_orders) * dof, p_dim), dtype=float)

    for block_index, deriv_order in enumerate(derivative_orders):
        traj = trajectory_derivative_maps.get(int(deriv_order), None)
        if traj is None:
            continue
        q_r = np.asarray(traj.q_at(p_vec, k), dtype=float).reshape(-1)
        J_r = np.asarray(traj.dqdp_at(k), dtype=float)
        if q_r.size != dof:
            raise ValueError(
                f"{error_prefix}: derivative map q size mismatch. "
                f"order={deriv_order}, expected {dof}, got {q_r.size}."
            )
        if J_r.shape != (dof, p_dim):
            raise ValueError(
                f"{error_prefix}: derivative map jacobian shape mismatch. "
                f"order={deriv_order}, expected {(dof, p_dim)}, got {J_r.shape}."
            )
        s = int(block_index * dof)
        e = int(s + dof)
        motion[s:e] = q_r
        dmotion_dp[s:e, :] = J_r

    return motion, dmotion_dp


def unique_jacobian_chain_candidates(candidates: list[Array]) -> tuple[Array, ...]:
    out: list[Array] = []
    seen_shapes: set[tuple[int, int]] = set()
    for cand_raw in candidates:
        cand = np.asarray(cand_raw, dtype=float)
        shape = (int(cand.shape[0]), int(cand.shape[1]))
        if shape in seen_shapes:
            continue
        out.append(cand)
        seen_shapes.add(shape)
    return tuple(out)


def chain_param_jacobian(
    J_raw: Array,
    *,
    q_var: str,
    state_jacobian_var: str,
    key: StateKey,
    jacobian_wrt: str | None,
    dqdp: Array,
    dmotiondp: Array,
    error_prefix: str,
    extra_candidates: tuple[Array, ...] = (),
) -> Array:
    J = np.asarray(J_raw, dtype=float)
    if J.ndim != 2:
        raise ValueError(f"{error_prefix}: Jacobian must be 2D, got shape {J.shape} for key {key!r}.")

    wrt = None if jacobian_wrt is None else str(jacobian_wrt)
    if wrt == str(q_var):
        return J

    if wrt == str(state_jacobian_var):
        candidates = unique_jacobian_chain_candidates(
            [*extra_candidates, np.asarray(dmotiondp, dtype=float), np.asarray(dqdp, dtype=float)]
        )
        for candidate in candidates:
            if J.shape[1] == candidate.shape[0]:
                return J @ candidate
        raise ValueError(
            f"{error_prefix}: Jacobian chain mismatch. "
            f"J_raw has shape {J.shape}, dqdp has shape {np.asarray(dqdp).shape}, "
            f"dmotiondp has shape {np.asarray(dmotiondp).shape}."
        )

    raise ValueError(
        f"{error_prefix}: unsupported jacobian_wrt metadata for parameter chain. "
        f"Expected {q_var!r} or {state_jacobian_var!r}, got {wrt!r}."
    )


__all__ = [
    "accept_trajectory_state_key",
    "chain_param_jacobian",
    "compose_interleaved_motion_and_jac",
    "compose_stacked_motion_and_jac",
    "expected_trajectory_steps",
    "is_param_jacobian_key",
    "TrajectoryStateBuilderMixin",
    "unique_jacobian_chain_candidates",
    "validate_trajectory_derivative_maps",
]
