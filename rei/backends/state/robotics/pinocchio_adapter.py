from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ....core.state_cache import StateKey
from ....core.state_schema import DTYPE_COORD, DTYPE_DYNAMICS
from ...optional import import_optional_backend, require_module_attrs
from .spatial import Jacobian6Order, linear_part_from_jacobian6

Array = np.ndarray

pin = import_optional_backend(
    "pinocchio",
    backend_name="rei.backends.state.robotics.pinocchio",
    install_hint="uv sync --group pinocchio",
)

_REQUIRED_ATTRS = ("buildModelFromUrdf", "computeFrameJacobian", "forwardKinematics", "updateFramePlacements")
require_module_attrs(
    pin,
    _REQUIRED_ATTRS,
    backend_name="rei.backends.state.robotics.pinocchio",
    install_hint="uv sync --group pinocchio",
    extra_hint="You may have installed a different PyPI package named `pinocchio`.",
)


def compute_pinocchio_frame_jacobian(model: Any, data: Any, q: Array, frame_ref: Any) -> Array:
    """Pinocchio-specific `computeFrameJacobian` wrapper with API/version fallbacks."""

    frame_id = int(frame_ref)
    ref = getattr(pin, "ReferenceFrame", None)
    for name in ("LOCAL_WORLD_ALIGNED", "LOCAL"):
        rf = getattr(ref, name, None) if ref is not None else None
        if rf is None:
            continue
        try:
            return pin.computeFrameJacobian(model, data, q, frame_id, rf)
        except TypeError:
            break
    return pin.computeFrameJacobian(model, data, q, frame_id)


def _as_dyn_vec(x: Any) -> Array:
    if hasattr(x, "vector"):
        return np.asarray(getattr(x, "vector"), dtype=float).reshape(-1)
    return np.asarray(x, dtype=float).reshape(-1)


class PinocchioAdapter:
    """Pinocchio-specific state handlers used by `PinocchioStateBuilder`."""

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        owner_type: str = "link",
        dynamics_owner_type: str = "total_joint",
        jac6_order: Jacobian6Order = "linear_angular",
        finite_diff_eps: float = 1e-8,
        torque_jacobian: str = "auto",
    ) -> None:
        self.model = model
        self.data = data
        self.owner_type = str(owner_type)
        self.dynamics_owner_type = str(dynamics_owner_type)
        self.jac6_order = jac6_order
        self.finite_diff_eps = float(finite_diff_eps)
        self.torque_jacobian = str(torque_jacobian)
        self.motion_provider: Callable[[Array, StateKey], tuple[Array, Array, Array]] | None = None

    def update(self, q: Array) -> None:
        pin.forwardKinematics(self.model, self.data, q)
        if hasattr(pin, "computeJointJacobians"):
            pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def resolve_ref(self, key: StateKey) -> Any:
        owner = getattr(key, "owner", None)
        owner_type = getattr(owner, "owner_type", None)
        owner_name = getattr(owner, "owner_name", None)
        if not isinstance(owner_name, str) or owner_name == "":
            raise ValueError(f"Pinocchio backend expects non-empty owner_name in key, got: {key!r}")

        if owner_type == self.owner_type:
            return int(self.model.getFrameId(owner_name))

        if owner_type == self.dynamics_owner_type and getattr(key, "dtype", None) == DTYPE_DYNAMICS:
            return ("dynamics", owner_name)

        if (
            owner_type == "total_joint"
            and getattr(key, "dtype", None) == DTYPE_COORD
            and str(getattr(key, "field", "")) == "q"
        ):
            return ("total_joint", owner_name, "q")

        raise ValueError(
            "Pinocchio backend expects owner_type="
            f"{self.owner_type!r} (kinematics) or {self.dynamics_owner_type!r} (dynamics), got: {key!r}"
        )

    def frame_pos(self, frame_ref: Any) -> Array:
        frame_id = int(frame_ref)
        return np.asarray(self.data.oMf[frame_id].translation, dtype=float).reshape(3)

    def frame_rot(self, frame_ref: Any) -> Array:
        frame_id = int(frame_ref)
        rot = np.asarray(self.data.oMf[frame_id].rotation, dtype=float).reshape(3, 3)
        return rot.reshape(-1)

    def frame_value(self, field: str, frame_ref: Any) -> Array:
        if field == "pos":
            return self.frame_pos(frame_ref)
        if field == "rot":
            return self.frame_rot(frame_ref)
        if field == "frame":
            return np.concatenate([self.frame_pos(frame_ref), self.frame_rot(frame_ref)], axis=0)
        raise ValueError(f"PinocchioAdapter: unsupported value field: {field!r}")

    def finite_difference_jacobian(
        self,
        q: Array,
        *,
        frame_ref: Any,
        value_fn: Callable[[Any], Array],
    ) -> Array:
        q0 = np.asarray(q, dtype=float).reshape(-1)
        y0 = np.asarray(value_fn(frame_ref), dtype=float).reshape(-1)
        m = int(y0.size)
        n = int(q0.size)
        J = np.zeros((m, n), dtype=float)
        if n == 0:
            return J

        eps = float(self.finite_diff_eps)
        try:
            for i in range(n):
                h = eps * max(1.0, abs(float(q0[i])))
                q_plus = q0.copy()
                q_plus[i] += h
                self.update(q_plus)
                y_plus = np.asarray(value_fn(frame_ref), dtype=float).reshape(-1)
                J[:, i] = (y_plus - y0) / h
        finally:
            self.update(q0)

        return J

    def pos(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del q, key
        return self.frame_pos(frame_ref)

    def pos_jac(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del key
        J6 = compute_pinocchio_frame_jacobian(self.model, self.data, q, frame_ref)
        return linear_part_from_jacobian6(J6, order=self.jac6_order)

    def rot(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del q, key
        return self.frame_rot(frame_ref)

    def rot_jac(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del key
        return self.finite_difference_jacobian(
            q,
            frame_ref=frame_ref,
            value_fn=self.frame_rot,
        )

    def frame(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del q, key
        return self.frame_value("frame", frame_ref)

    def frame_jac(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del key
        return self.finite_difference_jacobian(
            q,
            frame_ref=frame_ref,
            value_fn=lambda ref: self.frame_value("frame", ref),
        )

    def pin_nv(self, *, default: int) -> int:
        raw = getattr(self.model, "nv", None)
        if raw is None:
            return int(default)
        try:
            return int(raw)
        except Exception:
            return int(default)

    def motion_triplet(self, q: Array, key: StateKey) -> tuple[Array, Array, Array]:
        if self.motion_provider is not None:
            return self.motion_provider(q, key)
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        nv = self.pin_nv(default=int(q_vec.size))
        return q_vec, np.zeros((nv,), dtype=float), np.zeros((nv,), dtype=float)

    def torque_value_with_motion(
        self,
        *,
        q_vec: Array,
        v_vec: Array,
        a_vec: Array,
    ) -> Array:
        q_use = np.asarray(q_vec, dtype=float).reshape(-1)
        nv = self.pin_nv(default=int(q_use.size))
        v_use = np.asarray(v_vec, dtype=float).reshape(-1)
        a_use = np.asarray(a_vec, dtype=float).reshape(-1)
        if v_use.size != nv:
            raise ValueError(
                "PinocchioAdapter: velocity size mismatch for torque computation. "
                f"Expected nv={nv}, got {v_use.size}."
            )
        if a_use.size != nv:
            raise ValueError(
                "PinocchioAdapter: acceleration size mismatch for torque computation. "
                f"Expected nv={nv}, got {a_use.size}."
            )

        if hasattr(pin, "rnea"):
            tau = pin.rnea(self.model, self.data, q_use, v_use, a_use)
            return _as_dyn_vec(tau)

        if hasattr(pin, "computeGeneralizedGravity"):
            tau = pin.computeGeneralizedGravity(self.model, self.data, q_use)
            return _as_dyn_vec(tau)

        raise ValueError("PinocchioAdapter: torque computation requires rnea or computeGeneralizedGravity.")

    def torque_value(self, q: Array) -> Array:
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        nv = self.pin_nv(default=int(q_vec.size))
        return self.torque_value_with_motion(
            q_vec=q_vec,
            v_vec=np.zeros((nv,), dtype=float),
            a_vec=np.zeros((nv,), dtype=float),
        )

    def torque(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del state_ref
        q_use, dq_use, ddq_use = self.motion_triplet(q, key)
        return self.torque_value_with_motion(
            q_vec=q_use,
            v_vec=dq_use,
            a_vec=ddq_use,
        )

    def torque_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del state_ref
        q_use, dq_use, ddq_use = self.motion_triplet(q, key)
        J_motion = self.torque_motion_jacobian(
            q_vec=q_use,
            v_vec=dq_use,
            a_vec=ddq_use,
        )
        if J_motion is not None:
            if self.motion_provider is not None:
                return J_motion
            q_size = int(np.asarray(q, dtype=float).reshape(-1).size)
            return J_motion[:, :q_size]

        if self.motion_provider is not None:
            dof = int(q_use.size)
            motion0 = np.concatenate([q_use, dq_use, ddq_use], axis=0)
            return self.finite_difference_motion_jacobian(motion0, dof=dof)
        return self.finite_difference_torque_jacobian(q)

    def torque_motion_jacobian(
        self,
        *,
        q_vec: Array,
        v_vec: Array,
        a_vec: Array,
    ) -> Array | None:
        mode = str(self.torque_jacobian)
        if mode == "finite_difference":
            return None
        if not hasattr(pin, "computeRNEADerivatives"):
            if mode == "analytic":
                raise ValueError(
                    "PinocchioAdapter: torque_jacobian='analytic' requires "
                    "pin.computeRNEADerivatives."
                )
            return None

        q_use = np.asarray(q_vec, dtype=float).reshape(-1)
        v_use = np.asarray(v_vec, dtype=float).reshape(-1)
        a_use = np.asarray(a_vec, dtype=float).reshape(-1)
        out = pin.computeRNEADerivatives(self.model, self.data, q_use, v_use, a_use)
        if out is None:
            parts_raw = (
                getattr(self.data, "dtau_dq", None),
                getattr(self.data, "dtau_dv", None),
                getattr(self.data, "dtau_da", None),
            )
        else:
            parts_raw = tuple(out)
        if len(parts_raw) < 3:
            raise ValueError(
                "PinocchioAdapter: computeRNEADerivatives must provide "
                "(dtau_dq, dtau_dv, dtau_da)."
            )
        parts = [np.asarray(p, dtype=float) for p in parts_raw[:3]]
        if any(p.ndim != 2 for p in parts):
            shapes = [p.shape for p in parts]
            raise ValueError(
                "PinocchioAdapter: RNEA derivative blocks must be 2D, "
                f"got {shapes}."
            )
        return np.hstack(parts)

    def finite_difference_torque_jacobian(self, q: Array) -> Array:
        q0 = np.asarray(q, dtype=float).reshape(-1)
        y0 = self.torque_value(q0)
        m = int(y0.size)
        n = int(q0.size)
        J = np.zeros((m, n), dtype=float)
        if n == 0:
            return J

        eps = float(self.finite_diff_eps)
        for i in range(n):
            h = eps * max(1.0, abs(float(q0[i])))
            qp = q0.copy()
            qp[i] += h
            yp = self.torque_value(qp)
            J[:, i] = (yp - y0) / h
        return J

    def torque_from_motion(self, motion: Array, *, dof: int) -> Array:
        m = np.asarray(motion, dtype=float).reshape(-1)
        if m.size != 3 * int(dof):
            raise ValueError(
                "PinocchioAdapter: motion size mismatch for torque computation. "
                f"Expected {3 * int(dof)}, got {m.size}."
            )
        q_use = m[0:dof]
        dq_use = m[dof : 2 * dof]
        ddq_use = m[2 * dof : 3 * dof]
        return self.torque_value_with_motion(
            q_vec=q_use,
            v_vec=dq_use,
            a_vec=ddq_use,
        )

    def finite_difference_motion_jacobian(self, motion0: Array, *, dof: int) -> Array:
        y0 = self.torque_from_motion(motion0, dof=dof)
        m = int(y0.size)
        n = int(np.asarray(motion0, dtype=float).reshape(-1).size)
        J = np.zeros((m, n), dtype=float)
        if n == 0:
            return J

        eps = float(self.finite_diff_eps)
        for i in range(n):
            h = eps * max(1.0, abs(float(motion0[i])))
            mp = motion0.copy()
            mp[i] += h
            yp = self.torque_from_motion(mp, dof=dof)
            J[:, i] = (yp - y0) / h
        return J

    def momentum_value(self, q: Array) -> Array:
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        nv = int(getattr(self.model, "nv", q_vec.size))
        v0 = np.zeros((nv,), dtype=float)

        if hasattr(pin, "computeCentroidalMomentum"):
            h = pin.computeCentroidalMomentum(self.model, self.data, q_vec, v0)
            return _as_dyn_vec(h)

        if hasattr(pin, "ccrba"):
            h = pin.ccrba(self.model, self.data, q_vec, v0)
            if h is None:
                h = getattr(self.data, "hg", None)
            if h is None:
                raise ValueError("PinocchioAdapter: ccrba did not provide centroidal momentum.")
            return _as_dyn_vec(h)

        raise ValueError("PinocchioAdapter: momentum computation requires computeCentroidalMomentum or ccrba.")

    def momentum(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        return self.momentum_value(q)

    def momentum_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        q0 = np.asarray(q, dtype=float).reshape(-1)
        y0 = self.momentum_value(q0)
        m = int(y0.size)
        n = int(q0.size)
        J = np.zeros((m, n), dtype=float)
        if n == 0:
            return J

        eps = float(self.finite_diff_eps)
        for i in range(n):
            h = eps * max(1.0, abs(float(q0[i])))
            qp = q0.copy()
            qp[i] += h
            yp = self.momentum_value(qp)
            J[:, i] = (yp - y0) / h
        return J

    def force(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        return self.torque(q, key, state_ref)

    def force_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        return self.torque_jac(q, key, state_ref)


__all__ = [
    "compute_pinocchio_frame_jacobian",
    "pin",
    "PinocchioAdapter",
]
