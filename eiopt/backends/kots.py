from __future__ import annotations

from typing import Any

import numpy as np

from ..core.state_cache import StateKey
from ._template import BackendSingleFieldStateBuilder

try:
    from robokots.core.state import StateType
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "`eiopt.backends.kots` requires the robotics RoboKots bindings. "
        "Install RoboKots (e.g. via github) and retry."
    ) from e

Array = np.ndarray


class KotsFramePosStateBuilder(BackendSingleFieldStateBuilder):
    """RoboKots/Kots -> `build_state()` bridge for `dtype="kinematics", field="pos"` keys.

    This module intentionally delegates all common logic to
    `eiopt.backends._template.BackendSingleFieldStateBuilder`.

    You typically need to adjust only:
      - `_update_kinematics()` (how to run FK / prerequisite updates for the current q)
      - `_state_value()` (how to read a state value)
      - `_state_jacobian()` (how to compute the Jacobian)
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        q_var: str = "q",
    ) -> None:
        super().__init__(
            model,
            data,
            q_var=q_var,
            dtype="kinematics",
            owner_type="link",
            field="pos",
        )

    def _update_kinematics(self, q: Array) -> None:
        self.model.import_motions(np.asarray(q, dtype=float).reshape(-1))
        self.model.kinematics()

    def _resolve_state_ref(self, key: StateKey) -> Any:
        owner = getattr(key, "owner", None)
        owner_type = getattr(owner, "owner_type", None)
        owner_name = getattr(owner, "owner_name", None)
        if owner_type != self.owner_type or not isinstance(owner_name, str) or owner_name == "":
            raise ValueError(
                f"Kots backend expects owner_type={self.owner_type!r} in key, got: {key!r}"
            )
        frame_name = getattr(key, "frame", None) or "world"
        return StateType(self.owner_type, owner_name, self.field, str(frame_name))

    def _state_value(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del q, key
        return np.asarray(self.model.state_info(frame_ref), dtype=float).reshape(3)

    def _state_jacobian(self, q: Array, key: StateKey, frame_ref: Any) -> Array:
        del key
        del q
        J = np.asarray(self.model.jacobian(frame_ref), dtype=float)
        if J.ndim != 2:
            raise ValueError(f"Kots Jacobian must be 2D, got shape {J.shape}.")
        if J.shape[0] == 3:
            return J
        if J.shape[1] == 3:
            return J.T
        raise ValueError(f"Kots Jacobian must be (3,n) or (n,3), got {J.shape}.")
