from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.state_cache import StateKey
from ..core.state_schema import DTYPE_DYNAMICS, DTYPE_KINEMATICS, canonical_field_name
from ._template import BackendDispatchStateBuilder
from ._spatial import Jacobian6Order, linear_part_from_jacobian6

try:
    import pinocchio as pin
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "`eiopt.backends.pinocchio` requires the robotics Pinocchio bindings. "
        "Install Pinocchio (e.g. via conda-forge) and retry."
    ) from e

_REQUIRED_ATTRS = ("buildModelFromUrdf", "computeFrameJacobian", "forwardKinematics", "updateFramePlacements")
_missing = [a for a in _REQUIRED_ATTRS if not hasattr(pin, a)]
if _missing:  # pragma: no cover
    raise ImportError(
        "The imported `pinocchio` module is missing required robotics APIs: "
        f"{', '.join(_missing)}.\n"
        "You may have installed a different PyPI package named `pinocchio` (often version 0.1).\n"
        "Uninstall it and install the robotics Pinocchio bindings (typically via conda-forge)."
    )

Array = np.ndarray


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


@dataclass(frozen=True)
class PinocchioFieldFamily:
    field: str
    value_handler_name: str
    jac_handler_name: str


# pinocchio.py 内で「どの field ファミリを提供するか」を宣言する登録リスト。
# StateKey の (dtype, owner_type, field) マッチは BackendDispatchStateBuilder 側で自動実行される。
PINOCCHIO_DEFAULT_FIELD_FAMILIES: tuple[PinocchioFieldFamily, ...] = (
    PinocchioFieldFamily(field="pos", value_handler_name="_handle_pos", jac_handler_name="_handle_pos_jac"),
    PinocchioFieldFamily(field="rot", value_handler_name="_handle_rot", jac_handler_name="_handle_rot_jac"),
    PinocchioFieldFamily(field="frame", value_handler_name="_handle_frame", jac_handler_name="_handle_frame_jac"),
)


class PinocchioStateBuilder(BackendDispatchStateBuilder):
    """Pinocchio -> `build_state()` bridge with StateKey-based automatic dispatch.

    `PINOCCHIO_DEFAULT_FIELD_FAMILIES` をもとに、
    `(dtype="kinematics", owner_type="link", field=<...>)` の handler を一括登録する。
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        q_var: str = "q",
        jac6_order: Jacobian6Order = "linear_angular",
        finite_diff_eps: float = 1e-8,
        fields: Sequence[str] | None = None,
        dynamics_fields: Sequence[str] | None = None,
        dynamics_owner_type: str = "total_joint",
        dynamics_custom_handlers: Mapping[str, tuple[Callable[..., Array], Callable[..., Array]]] | None = None,
    ) -> None:
        super().__init__(model, data, q_var=q_var)
        self.dtype = DTYPE_KINEMATICS
        self.owner_type = "link"
        self.dynamics_owner_type = str(dynamics_owner_type)
        if self.dynamics_owner_type == "":
            raise ValueError("PinocchioStateBuilder: dynamics_owner_type must be non-empty.")
        self.jac6_order = jac6_order
        self.finite_diff_eps = float(finite_diff_eps)
        if self.finite_diff_eps <= 0.0:
            raise ValueError("PinocchioStateBuilder: finite_diff_eps must be > 0.")

        family_map = {spec.field: spec for spec in PINOCCHIO_DEFAULT_FIELD_FAMILIES}
        selected_fields = [spec.field for spec in PINOCCHIO_DEFAULT_FIELD_FAMILIES] if fields is None else [str(f) for f in fields]
        if len(selected_fields) == 0:
            raise ValueError("PinocchioStateBuilder: fields must be non-empty.")

        self.field_to_jac: dict[str, str] = {}
        for field in selected_fields:
            spec = family_map.get(field, None)
            if spec is None:
                supported = ", ".join(sorted(family_map.keys()))
                raise ValueError(
                    f"PinocchioStateBuilder: unsupported field {field!r}. "
                    f"Supported fields: {supported}."
                )
            value_handler = getattr(self, spec.value_handler_name)
            jac_handler = getattr(self, spec.jac_handler_name)
            _value_name, jac_name = self.register_value_and_jac(
                dtype=self.dtype,
                owner_type=self.owner_type,
                field=spec.field,
                value_handler=value_handler,
                jac_handler=jac_handler,
            )
            self.field_to_jac[spec.field] = jac_name

        if dynamics_fields is not None:
            dyn_fields = [canonical_field_name(str(f)) for f in dynamics_fields]
            dyn_fields = list(dict.fromkeys(dyn_fields))
            if len(dyn_fields) == 0:
                raise ValueError("PinocchioStateBuilder: dynamics_fields must be non-empty when provided.")
            for field in dyn_fields:
                if field == "torque":
                    self.register_value_and_jac(
                        dtype=DTYPE_DYNAMICS,
                        owner_type=self.dynamics_owner_type,
                        field="torque",
                        value_handler=self._handle_torque,
                        jac_handler=self._handle_torque_jac,
                    )
                    continue
                if field == "momentum":
                    self.register_value_and_jac(
                        dtype=DTYPE_DYNAMICS,
                        owner_type=self.dynamics_owner_type,
                        field="momentum",
                        value_handler=self._handle_momentum,
                        jac_handler=self._handle_momentum_jac,
                    )
                    continue
                if field == "force":
                    # Joint-space generalized force alias. In this backend it maps to torque.
                    self.register_value_and_jac(
                        dtype=DTYPE_DYNAMICS,
                        owner_type=self.dynamics_owner_type,
                        field="force",
                        value_handler=self._handle_force,
                        jac_handler=self._handle_force_jac,
                    )
                    continue
                if dynamics_custom_handlers is not None and field in dynamics_custom_handlers:
                    value_handler, jac_handler = dynamics_custom_handlers[field]
                    self.register_value_and_jac(
                        dtype=DTYPE_DYNAMICS,
                        owner_type=self.dynamics_owner_type,
                        field=field,
                        value_handler=value_handler,
                        jac_handler=jac_handler,
                    )
                    continue
                raise ValueError(
                    f"PinocchioStateBuilder: unsupported dynamics field {field!r}. "
                    "Currently supported: 'torque', 'momentum', 'force' "
                    "(plus dynamics_custom_handlers)."
                )

    def _update_kinematics(self, q: Array) -> None:
        pin.forwardKinematics(self.model, self.data, q)
        if hasattr(pin, "computeJointJacobians"):
            pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def _resolve_state_ref(self, key: StateKey) -> Any:
        owner = getattr(key, "owner", None)
        owner_type = getattr(owner, "owner_type", None)
        owner_name = getattr(owner, "owner_name", None)
        if not isinstance(owner_name, str) or owner_name == "":
            raise ValueError(f"Pinocchio backend expects non-empty owner_name in key, got: {key!r}")

        if owner_type == self.owner_type:
            return int(self.model.getFrameId(owner_name))

        if owner_type == self.dynamics_owner_type and getattr(key, "dtype", None) == DTYPE_DYNAMICS:
            return ("dynamics", owner_name)

        raise ValueError(
            "Pinocchio backend expects owner_type="
            f"{self.owner_type!r} (kinematics) or {self.dynamics_owner_type!r} (dynamics), got: {key!r}"
        )

    def _frame_pos(self, frame_ref: Any) -> Array:
        frame_id = int(frame_ref)
        return np.asarray(self.data.oMf[frame_id].translation, dtype=float).reshape(3)

    def _frame_rot(self, frame_ref: Any) -> Array:
        frame_id = int(frame_ref)
        rot = np.asarray(self.data.oMf[frame_id].rotation, dtype=float).reshape(3, 3)
        return rot.reshape(-1)

    def _frame_value(self, field: str, frame_ref: Any) -> Array:
        if field == "pos":
            return self._frame_pos(frame_ref)
        if field == "rot":
            return self._frame_rot(frame_ref)
        if field == "frame":
            return np.concatenate([self._frame_pos(frame_ref), self._frame_rot(frame_ref)], axis=0)
        raise ValueError(f"PinocchioStateBuilder: unsupported value field: {field!r}")

    def _finite_difference_jacobian(
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
                self._update_kinematics(q_plus)
                y_plus = np.asarray(value_fn(frame_ref), dtype=float).reshape(-1)
                J[:, i] = (y_plus - y0) / h
        finally:
            self._update_kinematics(q0)

        return J

    def _handle_pos(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del q, key
        return self._frame_pos(frame_ref)

    def _handle_pos_jac(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del key
        J6 = compute_pinocchio_frame_jacobian(self.model, self.data, q, frame_ref)
        return linear_part_from_jacobian6(J6, order=self.jac6_order)

    def _handle_rot(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del q, key
        return self._frame_rot(frame_ref)

    def _handle_rot_jac(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del key
        return self._finite_difference_jacobian(
            q,
            frame_ref=frame_ref,
            value_fn=self._frame_rot,
        )

    def _handle_frame(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del q, key
        return self._frame_value("frame", frame_ref)

    def _handle_frame_jac(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del key
        return self._finite_difference_jacobian(
            q,
            frame_ref=frame_ref,
            value_fn=lambda ref: self._frame_value("frame", ref),
        )

    def _torque_value(self, q: Array) -> Array:
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        if hasattr(pin, "computeGeneralizedGravity"):
            tau = pin.computeGeneralizedGravity(self.model, self.data, q_vec)
            return _as_dyn_vec(tau)
        if hasattr(pin, "rnea"):
            nv = int(getattr(self.model, "nv", q_vec.size))
            v = np.zeros((nv,), dtype=float)
            a = np.zeros((nv,), dtype=float)
            tau = pin.rnea(self.model, self.data, q_vec, v, a)
            return _as_dyn_vec(tau)
        raise ValueError("PinocchioStateBuilder: torque computation requires computeGeneralizedGravity or rnea.")

    def _handle_torque(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        return self._torque_value(q)

    def _handle_torque_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key
        q0 = np.asarray(q, dtype=float).reshape(-1)
        y0 = self._torque_value(q0)
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
            yp = self._torque_value(qp)
            J[:, i] = (yp - y0) / h
        return J

    def _momentum_value(self, q: Array) -> Array:
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
                raise ValueError("PinocchioStateBuilder: ccrba did not provide centroidal momentum.")
            return _as_dyn_vec(h)

        raise ValueError("PinocchioStateBuilder: momentum computation requires computeCentroidalMomentum or ccrba.")

    def _handle_momentum(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        return self._momentum_value(q)

    def _handle_momentum_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        q0 = np.asarray(q, dtype=float).reshape(-1)
        y0 = self._momentum_value(q0)
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
            yp = self._momentum_value(qp)
            J[:, i] = (yp - y0) / h
        return J

    def _handle_force(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        return self._torque_value(q)

    def _handle_force_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        return self._handle_torque_jac(q, key, state_ref)


def _as_dyn_vec(x: Any) -> Array:
    if hasattr(x, "vector"):
        return np.asarray(getattr(x, "vector"), dtype=float).reshape(-1)
    return np.asarray(x, dtype=float).reshape(-1)
