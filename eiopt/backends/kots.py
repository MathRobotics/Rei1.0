from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.state_cache import StateKey
from ..core.state_schema import DTYPE_KINEMATICS
from ._template import BackendDispatchStateBuilder

try:
    from robokots.core.state import StateType
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "`eiopt.backends.kots` requires the robotics RoboKots bindings. "
        "Install RoboKots (e.g. via github) and retry."
    ) from e

Array = np.ndarray


@dataclass(frozen=True)
class KotsFieldFamily:
    field: str


# kots.py 内で「どの field ファミリを提供するか」を宣言する登録リスト。
KOTS_DEFAULT_FIELD_FAMILIES: tuple[KotsFieldFamily, ...] = (
    KotsFieldFamily(field="pos"),
    KotsFieldFamily(field="rot"),
    KotsFieldFamily(field="frame"),
)


class KotsStateBuilder(BackendDispatchStateBuilder):
    """RoboKots/Kots -> `build_state()` bridge with StateKey-based automatic dispatch."""

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        q_var: str = "q",
        fields: Sequence[str] | None = None,
    ) -> None:
        super().__init__(model, data, q_var=q_var)
        self.dtype = DTYPE_KINEMATICS
        self.owner_type = "link"

        family_map = {spec.field: spec for spec in KOTS_DEFAULT_FIELD_FAMILIES}
        selected_fields = [spec.field for spec in KOTS_DEFAULT_FIELD_FAMILIES] if fields is None else [str(f) for f in fields]
        if len(selected_fields) == 0:
            raise ValueError("KotsStateBuilder: fields must be non-empty.")

        self.field_to_jac: dict[str, str] = {}
        for field in selected_fields:
            spec = family_map.get(field, None)
            if spec is None:
                supported = ", ".join(sorted(family_map.keys()))
                raise ValueError(
                    f"KotsStateBuilder: unsupported field {field!r}. "
                    f"Supported fields: {supported}."
                )
            _value_name, jac_name = self.register_value_and_jac(
                dtype=self.dtype,
                owner_type=self.owner_type,
                field=spec.field,
                value_handler=self._handle_value,
                jac_handler=self._handle_jac,
            )
            self.field_to_jac[spec.field] = jac_name

    def _update_kinematics(self, q: Array) -> None:
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        dof = self._model_dof()
        order = self._model_order()

        if q_vec.size == dof * order:
            motion = q_vec
        elif q_vec.size == dof:
            motion = self._expand_coordinate_motion(q_vec, dof=dof, order=order)
        else:
            raise ValueError(
                "KotsStateBuilder: unexpected q size. "
                f"Expected dof ({dof}) or dof*order ({dof * order}), got {q_vec.size}."
            )

        self.model.import_motions(motion)
        self.model.kinematics()

    def _model_dof(self) -> int:
        dof_fn = getattr(self.model, "dof", None)
        if callable(dof_fn):
            return int(dof_fn())
        robot = getattr(self.model, "robot_", None)
        if robot is not None and hasattr(robot, "dof"):
            return int(getattr(robot, "dof"))
        raise ValueError("KotsStateBuilder: unable to resolve model dof.")

    def _model_order(self) -> int:
        order_fn = getattr(self.model, "order", None)
        if callable(order_fn):
            order = int(order_fn())
        else:
            order = int(getattr(self.model, "order_", 1))
        if order < 1:
            raise ValueError(f"KotsStateBuilder: model order must be >= 1, got {order}.")
        return order

    def _expand_coordinate_motion(self, q: Array, *, dof: int, order: int) -> Array:
        motion = np.zeros(dof * order, dtype=float)
        robot = getattr(self.model, "robot_", None)
        if robot is None:
            for i in range(min(q.size, dof)):
                motion[i * order] = float(q[i])
            return motion

        owners = [*getattr(robot, "links", []), *getattr(robot, "joints", [])]
        owners = [owner for owner in owners if int(getattr(owner, "dof", 0)) > 0]
        owners.sort(key=lambda owner: int(getattr(owner, "dof_index", 0)))

        cursor = 0
        for owner in owners:
            owner_dof = int(getattr(owner, "dof", 0))
            dof_index = int(getattr(owner, "dof_index", 0))
            start = dof_index * order
            stop = start + owner_dof
            if stop > motion.size:
                raise ValueError("KotsStateBuilder: invalid dof_index/dof in robot structure.")
            motion[start:stop] = q[cursor : cursor + owner_dof]
            cursor += owner_dof

        if cursor != q.size:
            raise ValueError(
                "KotsStateBuilder: failed to map q into motion coordinates. "
                f"Mapped {cursor} elements from q size {q.size}."
            )
        return motion

    def _resolve_state_ref(self, key: StateKey) -> Any:
        owner = getattr(key, "owner", None)
        owner_type = getattr(owner, "owner_type", None)
        owner_name = getattr(owner, "owner_name", None)
        if owner_type != self.owner_type or not isinstance(owner_name, str) or owner_name == "":
            raise ValueError(
                f"Kots backend expects owner_type={self.owner_type!r} in key, got: {key!r}"
            )
        frame_name = getattr(key, "frame", None) or "world"
        return StateType(self.owner_type, owner_name, key.field, str(frame_name))

    def _value_from_state_ref(self, state_ref: Any) -> Array:
        return np.asarray(self.model.state_info(state_ref), dtype=float).reshape(-1)

    def _handle_value(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del q, key
        return self._value_from_state_ref(state_ref)

    def _handle_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key
        del q
        J = np.asarray(self.model.jacobian(state_ref), dtype=float)
        if J.ndim != 2:
            raise ValueError(f"Kots Jacobian must be 2D, got shape {J.shape}.")

        m = int(self._value_from_state_ref(state_ref).size)
        if J.shape[0] == m:
            return J
        if J.shape[1] == m:
            return J.T
        raise ValueError(f"Kots Jacobian must be ({m},n) or (n,{m}), got {J.shape}.")
