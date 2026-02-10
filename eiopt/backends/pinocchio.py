from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.state_cache import StateKey
from ..core.state_schema import DTYPE_KINEMATICS
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
    ) -> None:
        super().__init__(model, data, q_var=q_var)
        self.dtype = DTYPE_KINEMATICS
        self.owner_type = "link"
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

    def _update_kinematics(self, q: Array) -> None:
        pin.forwardKinematics(self.model, self.data, q)
        if hasattr(pin, "computeJointJacobians"):
            pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def _resolve_state_ref(self, key: StateKey) -> Any:
        owner = getattr(key, "owner", None)
        owner_type = getattr(owner, "owner_type", None)
        owner_name = getattr(owner, "owner_name", None)
        if owner_type != self.owner_type or not isinstance(owner_name, str) or owner_name == "":
            raise ValueError(
                f"Pinocchio backend expects owner_type={self.owner_type!r} in key, got: {key!r}"
            )
        return int(self.model.getFrameId(owner_name))

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
