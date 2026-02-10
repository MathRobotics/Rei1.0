from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._spatial import Jacobian6Order, linear_part_from_jacobian6
from ._template import BackendFramePosStateBuilder

try:
    from robokots.kots import *
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "`eiopt.backends.robokots` requires the robotics RoboKots bindings. "
        "Install RoboKots (e.g. via github) and retry."
    ) from e

Array = np.ndarray

class KotsFramePosStateBuilder(BackendFramePosStateBuilder):
    """RoboKots/Kots -> `build_state()` bridge for `dtype="frame", field="pos"` keys.

    This module intentionally delegates all common logic to
    `eiopt.backends._template.BackendFramePosStateBuilder`.

    You typically need to adjust only:
      - `_update_kinematics()` (how to run FK / prerequisite updates for the current q)
      - `_frame_pos()` (how to read a frame position)
      - `_frame_pos_jacobian()` (how to compute the position Jacobian)
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        q_var: str = "q",
        jac6_order: Jacobian6Order = "angular_linear",
    ) -> None:
        super().__init__(model, data, q_var=q_var)
        self.jac6_order = jac6_order
        self.state = StateType('link','end','snap')

    def _update_kinematics(self, q: Array) -> None:
        self.model.import_motion(q)
        self.model.forward_kinematics()
        self.data.dict.append(self.model.state_dict())

    def _resolve_frame_id(self, frame_name: str) -> int:
        get_frame_id = getattr(self.model, "getFrameId", None)
        if callable(get_frame_id):
            return int(get_frame_id(str(frame_name)))
        raise NotImplementedError("Kots backend must implement frame name to id resolution.")

    def _frame_pos(self, frame_id: int) -> Array:
        self.data.dict[frame_id]
        return robotkos.outward.state.get_value(self.model.robot, self.state) 

    def _frame_pos_jacobian(self, q: Array, frame_id: int) -> Array:
        del q, frame_id
        return self.model.jacobian(self.state)
