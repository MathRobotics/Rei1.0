from __future__ import annotations

from typing import Any

import numpy as np

from ._template import BackendFramePosStateBuilder
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

def compute_pinocchio_frame_jacobian(model: Any, data: Any, q: Array, frame_id: int) -> Array:
    """Pinocchio-specific `computeFrameJacobian` wrapper with API/version fallbacks.

    Some Pinocchio versions expose `ReferenceFrame` and accept it as an extra argument,
    while others only support the 4-argument form. We prefer `LOCAL_WORLD_ALIGNED`
    (useful for position tasks) and fall back to `LOCAL`, then to the legacy call.
    """

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


class PinocchioFramePosStateBuilder(BackendFramePosStateBuilder):
    """Pinocchio -> `build_state()` bridge for `dtype="frame", field="pos"` keys.

    Supported keys (k=0 only):
      - dtype="frame", owner_type="link", field="pos"
      - dtype="frame", owner_type="link", field="pos_J_<q_var>"

    Notes:
      - This is intentionally minimal. Extend as needed (rot/frame/etc).
      - Decision variables (e.g. joint angles) are typically read via `get_var`
        rather than requested from `build_state()`.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        q_var: str = "q",
        jac6_order: Jacobian6Order = "linear_angular",
    ) -> None:
        super().__init__(model, data, q_var=q_var)
        self.jac6_order = jac6_order

    def _update_kinematics(self, q: Array) -> None:
        pin.forwardKinematics(self.model, self.data, q)
        if hasattr(pin, "computeJointJacobians"):
            pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def _resolve_frame_id(self, frame_name: str) -> int:
        return int(self.model.getFrameId(str(frame_name)))

    def _frame_pos(self, frame_id: int) -> Array:
        return self.data.oMf[frame_id].translation

    def _frame_pos_jacobian(self, q: Array, frame_id: int) -> Array:
        J6 = compute_pinocchio_frame_jacobian(self.model, self.data, q, frame_id)
        return linear_part_from_jacobian6(J6, order=self.jac6_order)
