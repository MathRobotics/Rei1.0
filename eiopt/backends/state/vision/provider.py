from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ....core.state_cache import StateKey
from ....core.state_schema import DTYPE_VISION
from ..dispatch.template import BackendDispatchStateBuilder, DispatchHandler

Array = np.ndarray
ModelUpdateFn = Callable[[Array, Any, Any], None]


@dataclass(frozen=True)
class VisionFieldHandler:
    value_handler: DispatchHandler
    jac_handler: DispatchHandler | None = None
    jacobian_wrt: str | None = None


class CameraCalibrationStateProvider(BackendDispatchStateBuilder):
    """Minimal `dtype="vision"` state provider template.

    This provider is intentionally generic and backend-agnostic:
      - input parameter vector is extracted from `param_var`
      - optional `update_model(q, model, data)` can sync model state
      - each vision field is registered via callback handlers
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        param_var: str = "theta",
        owner_type: str = "camera",
        fields: Sequence[str] | None = None,
        field_handlers: Mapping[str, VisionFieldHandler] | None = None,
        update_model: ModelUpdateFn | None = None,
        allow_nonzero_k: bool = True,
    ) -> None:
        super().__init__(
            model,
            data,
            q_var=param_var,
            allow_nonzero_k=allow_nonzero_k,
        )
        self.dtype = DTYPE_VISION
        self.owner_type = str(owner_type)
        if self.owner_type == "":
            raise ValueError("CameraCalibrationStateProvider: owner_type must be non-empty.")
        self.update_model = update_model
        self.field_to_jac: dict[str, str] = {}

        handlers = {} if field_handlers is None else {str(k): v for k, v in field_handlers.items()}
        selected_fields = list(handlers.keys()) if fields is None else [str(f) for f in fields]
        if len(selected_fields) == 0:
            raise ValueError(
                "CameraCalibrationStateProvider: no fields configured. "
                "Provide `field_handlers` (or explicit `fields` + matching handlers)."
            )

        for field in selected_fields:
            if field == "":
                raise ValueError("CameraCalibrationStateProvider: field names must be non-empty.")
            spec = handlers.get(field, None)
            if spec is None:
                raise ValueError(
                    "CameraCalibrationStateProvider: missing field handler for "
                    f"{field!r}. Provide it in `field_handlers`."
                )
            if not isinstance(spec, VisionFieldHandler):
                raise TypeError(
                    "CameraCalibrationStateProvider: field handler must be VisionFieldHandler. "
                    f"field={field!r}, got={type(spec).__name__!r}."
                )
            self.register_vision_field(
                field,
                value_handler=spec.value_handler,
                jac_handler=spec.jac_handler,
                jacobian_wrt=spec.jacobian_wrt,
            )

    def register_vision_field(
        self,
        field: str,
        *,
        value_handler: DispatchHandler,
        jac_handler: DispatchHandler | None = None,
        jacobian_wrt: str | None = None,
    ) -> tuple[str, str | None]:
        field_name = str(field)
        if field_name == "":
            raise ValueError("CameraCalibrationStateProvider: field must be non-empty.")
        if jac_handler is None:
            self.register_handler(
                dtype=self.dtype,
                owner_type=self.owner_type,
                field=field_name,
                handler=value_handler,
                state_ref_field=field_name,
            )
            return field_name, None

        _value_name, jac_name = self.register_value_and_jac(
            dtype=self.dtype,
            owner_type=self.owner_type,
            field=field_name,
            value_handler=value_handler,
            jac_handler=jac_handler,
            jacobian_wrt=jacobian_wrt,
        )
        self.field_to_jac[field_name] = jac_name
        return field_name, jac_name

    def _update_kinematics(self, q: Array) -> None:
        if self.update_model is None:
            return
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        self.update_model(q_vec, self.model, self.data)

    def _resolve_state_ref(self, key: StateKey) -> Any:
        owner = getattr(key, "owner", None)
        owner_type = getattr(owner, "owner_type", None)
        owner_name = getattr(owner, "owner_name", None)
        if not isinstance(owner_name, str) or owner_name == "":
            raise ValueError(
                "CameraCalibrationStateProvider expects non-empty owner_name in key, "
                f"got: {key!r}"
            )

        return {
            "k": int(getattr(key, "k", 0)),
            "owner_type": None if owner_type is None else str(owner_type),
            "owner_name": owner_name,
            "dtype": str(getattr(key, "dtype", "")),
            "field": str(getattr(key, "field", "")),
            "frame": getattr(key, "frame", None),
            "rel_frame": getattr(key, "rel_frame", None),
        }


__all__ = [
    "ModelUpdateFn",
    "VisionFieldHandler",
    "CameraCalibrationStateProvider",
]
