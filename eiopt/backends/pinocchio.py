from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import numpy as np

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

from ..core.state_cache import StateKey
from ..core.state_schema import DTYPE_FRAME, jac_field

Array = np.ndarray


def linear_part_from_frame_jacobian(J6: Array) -> Array:
    J6 = np.asarray(J6, dtype=float)
    if J6.ndim != 2:
        raise ValueError(f"Expected 2D Jacobian, got {J6.shape}.")
    if J6.shape[0] == 6:
        return J6[:3, :]
    if J6.shape[1] == 6:
        return J6[:, :3].T
    raise ValueError(f"Unexpected frame Jacobian shape: {J6.shape} (expected (6,n) or (n,6)).")


def compute_frame_jacobian(model: Any, data: Any, q: Array, frame_id: int) -> Array:
    ref = getattr(pin, "ReferenceFrame", None)
    lwa_ref = getattr(ref, "LOCAL_WORLD_ALIGNED", None) if ref is not None else None
    local_ref = getattr(ref, "LOCAL", None) if ref is not None else None

    # For position objectives we typically want the Jacobian expressed at the frame origin,
    # aligned with the world axes. `LOCAL_WORLD_ALIGNED` matches that expectation.
    for rf in (lwa_ref, local_ref):
        if rf is None:
            continue
        try:
            return pin.computeFrameJacobian(model, data, q, frame_id, rf)
        except TypeError:
            pass

    return pin.computeFrameJacobian(model, data, q, frame_id)


class PinocchioFramePosStateBuilder:
    """Pinocchio -> `build_state()` bridge for `dtype="frame", field="pos"` keys.

    Supported keys (k=0 only):
      - dtype="frame", owner_type="link", field="pos"
      - dtype="frame", owner_type="link", field="pos_J_<q_var>"

    Notes:
      - This is intentionally minimal. Extend as needed (rot/frame/etc).
      - Decision variables (e.g. joint angles) are typically read via `get_var`
        rather than requested from `build_state()`.
    """

    def __init__(self, model: Any, data: Any, *, q_var: str = "q") -> None:
        self.model = model
        self.data = data
        self.q_var = str(q_var)
        self._frame_id_cache: dict[str, int] = {}

    def _frame_id(self, frame_name: str) -> int:
        name = str(frame_name)
        if name in self._frame_id_cache:
            return self._frame_id_cache[name]
        frame_id = int(self.model.getFrameId(name))
        self._frame_id_cache[name] = frame_id
        return frame_id

    def build_state(
        self,
        x_all: Array,
        *,
        pack: Any = None,
        time: Any = None,
        required: Optional[Iterable[StateKey]] = None,
    ) -> dict[StateKey, Any]:
        del time

        if required is None:
            return {}

        x_all = np.asarray(x_all, dtype=float).reshape(-1)

        q = x_all
        if pack is not None and hasattr(pack, "slices") and self.q_var in getattr(pack, "slices", {}):
            s, e = pack.slices[self.q_var]
            q = np.asarray(x_all[s:e], dtype=float).reshape(-1)

        pin.forwardKinematics(self.model, self.data, q)
        if hasattr(pin, "computeJointJacobians"):
            pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        pos_field = "pos"
        jac_pos_field = jac_field(pos_field, var=self.q_var)

        needs: dict[str, tuple[bool, bool]] = {}
        for key in required:
            if not isinstance(key, StateKey):
                continue
            if int(getattr(key, "k", 0)) != 0:
                continue
            if getattr(key, "dtype", None) != DTYPE_FRAME:
                continue
            owner = getattr(key, "owner", None)
            if getattr(owner, "owner_type", None) != "link":
                continue
            frame_name = getattr(owner, "owner_name", None)
            if not isinstance(frame_name, str) or frame_name == "":
                continue

            need_pos, need_jac = needs.get(frame_name, (False, False))
            if key.field == pos_field:
                need_pos = True
            elif key.field == jac_pos_field:
                need_jac = True
            needs[frame_name] = (need_pos, need_jac)

        if not needs:
            return {}

        pos_by_frame: dict[str, Array] = {}
        Jpos_by_frame: dict[str, Array] = {}

        for frame_name, (need_pos, need_jac) in needs.items():
            frame_id = self._frame_id(frame_name)

            if need_pos:
                pose = self.data.oMf[frame_id]
                pos_by_frame[frame_name] = np.asarray(pose.translation, dtype=float).reshape(3).copy()

            if need_jac:
                J6 = compute_frame_jacobian(self.model, self.data, q, frame_id)
                Jpos_by_frame[frame_name] = linear_part_from_frame_jacobian(np.asarray(J6, dtype=float)).copy()

        out: dict[StateKey, Any] = {}
        for key in required:
            if not isinstance(key, StateKey):
                continue
            if int(getattr(key, "k", 0)) != 0:
                continue
            if getattr(key, "dtype", None) != DTYPE_FRAME:
                continue
            owner = getattr(key, "owner", None)
            if getattr(owner, "owner_type", None) != "link":
                continue
            frame_name = getattr(owner, "owner_name", None)
            if frame_name not in needs:
                continue

            if key.field == pos_field and frame_name in pos_by_frame:
                out[key] = pos_by_frame[frame_name]
            elif key.field == jac_pos_field and frame_name in Jpos_by_frame:
                out[key] = Jpos_by_frame[frame_name]

        return out
