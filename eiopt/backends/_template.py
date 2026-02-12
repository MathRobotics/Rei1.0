from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..core.state_cache import StateKey
from ..core.state_schema import jac_field

Array = np.ndarray
DispatchHandler = Callable[[Array, StateKey, Any], Any]


@dataclass(frozen=True)
class _DispatchEntry:
    handler: DispatchHandler
    state_ref_field: str | None = None
    jacobian_wrt: str | None = None


class BackendDispatchStateBuilder:
    """Template backend -> `build_state()` bridge using key-based dispatch.

    Typical usage:
      - implement `_update_kinematics()` / `_resolve_state_ref()`
      - register handlers with `register_handler()`
      - or use `register_value_and_jac()` to register a field pair in one call
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        q_var: str = "q",
    ) -> None:
        self.model = model
        self.data = data
        self.q_var = str(q_var)
        if self.q_var == "":
            raise ValueError("BackendDispatchStateBuilder: q_var must be non-empty.")
        self._dispatch: dict[tuple[str, str, str], _DispatchEntry] = {}
        self._state_ref_cache: dict[tuple[int, str, str, str, str, str | None, str | None], Any] = {}

    def _update_kinematics(self, q: Array) -> None:
        """Run backend FK/Jacobian prerequisites for the current `q`."""

        del q

    def _resolve_state_ref(self, key: StateKey) -> Any:
        """Resolve backend-specific state reference from a requested StateKey."""

        raise NotImplementedError("TODO: implement backend state reference resolution.")

    def register_handler(
        self,
        *,
        dtype: str,
        owner_type: str,
        field: str,
        handler: DispatchHandler,
        state_ref_field: str | None = None,
        jacobian_wrt: str | None = None,
    ) -> None:
        route = (str(dtype), str(owner_type), str(field))
        if any(part == "" for part in route):
            raise ValueError("BackendDispatchStateBuilder: dtype/owner_type/field must be non-empty.")
        if not callable(handler):
            raise TypeError("BackendDispatchStateBuilder: handler must be callable.")
        if state_ref_field is not None and str(state_ref_field) == "":
            raise ValueError("BackendDispatchStateBuilder: state_ref_field must be non-empty.")
        if jacobian_wrt is not None and str(jacobian_wrt) == "":
            raise ValueError("BackendDispatchStateBuilder: jacobian_wrt must be non-empty.")
        if route in self._dispatch:
            raise ValueError(f"BackendDispatchStateBuilder: duplicate handler route: {route}.")
        self._dispatch[route] = _DispatchEntry(
            handler=handler,
            state_ref_field=None if state_ref_field is None else str(state_ref_field),
            jacobian_wrt=None if jacobian_wrt is None else str(jacobian_wrt),
        )

    def register_handlers(
        self,
        *,
        dtype: str,
        owner_type: str,
        handlers: Mapping[str, DispatchHandler],
    ) -> None:
        for field, handler in handlers.items():
            self.register_handler(
                dtype=dtype,
                owner_type=owner_type,
                field=str(field),
                handler=handler,
            )

    def register_value_and_jac(
        self,
        *,
        dtype: str,
        owner_type: str,
        field: str,
        value_handler: DispatchHandler | None = None,
        jac_handler: DispatchHandler | None = None,
        jac_var: str | None = None,
        jacobian_wrt: str | None = None,
    ) -> tuple[str, str]:
        """Register value/jacobian handlers for one logical field family."""

        field_name = str(field)
        if field_name == "":
            raise ValueError("BackendDispatchStateBuilder: field must be non-empty.")
        if value_handler is None and jac_handler is None:
            raise ValueError("BackendDispatchStateBuilder: value_handler and jac_handler cannot both be None.")

        var = self.q_var if jac_var is None else str(jac_var)
        if var == "":
            raise ValueError("BackendDispatchStateBuilder: jac_var must be non-empty.")
        jac_name = jac_field(field_name, var=var)
        jacobian_wrt_name = var if jacobian_wrt is None else str(jacobian_wrt)
        if jacobian_wrt_name == "":
            raise ValueError("BackendDispatchStateBuilder: jacobian_wrt must be non-empty.")

        if value_handler is not None:
            self.register_handler(
                dtype=dtype,
                owner_type=owner_type,
                field=field_name,
                handler=value_handler,
                state_ref_field=field_name,
            )
        if jac_handler is not None:
            self.register_handler(
                dtype=dtype,
                owner_type=owner_type,
                field=jac_name,
                handler=jac_handler,
                state_ref_field=field_name,
                jacobian_wrt=jacobian_wrt_name,
            )
        return field_name, jac_name

    def _accept_required_key(self, key: StateKey) -> bool:
        if not isinstance(key, StateKey):
            return False
        if int(getattr(key, "k", 0)) != 0:
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

    def _route_for_key(self, key: StateKey) -> tuple[str, str, str] | None:
        owner = getattr(key, "owner", None)
        owner_type = getattr(owner, "owner_type", None)
        dtype = getattr(key, "dtype", None)
        field = getattr(key, "field", None)
        if not isinstance(owner_type, str) or not isinstance(dtype, str) or not isinstance(field, str):
            return None
        return dtype, owner_type, field

    def _extract_q(self, x_all: Array, *, pack: Any = None) -> Array:
        q = np.asarray(x_all, dtype=float).reshape(-1)
        if pack is not None and hasattr(pack, "slices") and self.q_var in getattr(pack, "slices", {}):
            s, e = pack.slices[self.q_var]
            q = np.asarray(q[s:e], dtype=float).reshape(-1)
        return q

    def _state_ref_query_key(self, key: StateKey, *, state_ref_field: str | None = None) -> StateKey:
        if state_ref_field is None or state_ref_field == key.field:
            return key
        return StateKey(
            k=int(key.k),
            owner=key.owner,
            dtype=key.dtype,
            field=state_ref_field,
            frame=getattr(key, "frame", None),
            rel_frame=getattr(key, "rel_frame", None),
        )

    def _state_ref_cache_key(self, key: StateKey, *, state_ref_field: str | None = None) -> tuple[int, str, str, str, str, str | None, str | None]:
        owner = getattr(key, "owner", None)
        owner_type = str(getattr(owner, "owner_type", ""))
        owner_name = str(getattr(owner, "owner_name", ""))
        dtype = str(getattr(key, "dtype", ""))
        field = str(getattr(key, "field", "")) if state_ref_field is None else str(state_ref_field)
        frame = getattr(key, "frame", None)
        rel_frame = getattr(key, "rel_frame", None)
        return (
            int(getattr(key, "k", 0)),
            owner_type,
            owner_name,
            dtype,
            field,
            None if frame is None else str(frame),
            None if rel_frame is None else str(rel_frame),
        )

    def _state_ref(self, key: StateKey, *, state_ref_field: str | None = None) -> Any:
        query_key = self._state_ref_query_key(key, state_ref_field=state_ref_field)
        cache_key = self._state_ref_cache_key(query_key, state_ref_field=state_ref_field)
        if cache_key in self._state_ref_cache:
            return self._state_ref_cache[cache_key]
        state_ref = self._resolve_state_ref(query_key)
        self._state_ref_cache[cache_key] = state_ref
        return state_ref

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

        q = self._extract_q(x_all, pack=pack)
        self._update_kinematics(q)

        out: dict[StateKey, Any] = {}
        for key in required:
            if not self._accept_required_key(key):
                continue
            route = self._route_for_key(key)
            if route is None:
                continue
            entry = self._dispatch.get(route, None)
            if entry is None:
                continue
            state_ref = self._state_ref(key, state_ref_field=entry.state_ref_field)
            out[key] = entry.handler(q, key, state_ref)
        return out
