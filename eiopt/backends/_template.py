from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import numpy as np

from ..core.state_cache import StateKey
from ..core.state_schema import DTYPE_KINEMATICS, jac_field

Array = np.ndarray


class BackendSingleFieldStateBuilder:
    """Template backend -> `build_state()` bridge for one `{dtype, owner_type, field}`.

    This helper computes values + Jacobians for one field family:
      - value field: `{field}`
      - jac field  : `{field}_J_{q_var}`
    for requested keys that match:
      - `k == 0`
      - `dtype == self.dtype`
      - `owner_type == self.owner_type`

    Subclasses only implement backend-specific parts:
      - `_update_kinematics()`
      - `_resolve_state_ref()`
      - `_state_value()`
      - `_state_jacobian()`

    Notes:
      - Keep this module importable without heavy dependencies.
        If you must import a backend package, wrap it in try/except.
      - `build_state()` should return only the keys requested in `required`.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        q_var: str = "q",
        dtype: str = DTYPE_KINEMATICS,
        owner_type: str = "link",
        field: str = "pos",
    ) -> None:
        self.model = model
        self.data = data
        self.q_var = str(q_var)
        self.dtype = str(dtype)
        self.owner_type = str(owner_type)
        self.field = str(field)
        if self.q_var == "":
            raise ValueError("BackendSingleFieldStateBuilder: q_var must be non-empty.")
        if self.field == "":
            raise ValueError("BackendSingleFieldStateBuilder: field must be non-empty.")
        self._state_ref_cache: dict[StateKey, Any] = {}

    def _update_kinematics(self, q: Array) -> None:
        """Run backend FK/Jacobian prerequisites for the current `q`."""

        raise NotImplementedError("TODO: implement backend kinematics update.")

    def _resolve_state_ref(self, key: StateKey) -> Any:
        """Resolve backend-specific state reference from a requested StateKey."""

        raise NotImplementedError("TODO: implement backend state reference resolution.")

    def _state_value(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        """Return state value for the canonical value key."""

        raise NotImplementedError("TODO: implement backend value extraction.")

    def _state_jacobian(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        """Return Jacobian for the canonical value key with respect to `q_var`."""

        raise NotImplementedError("TODO: implement backend Jacobian extraction.")

    def _accept_required_key(self, key: StateKey) -> bool:
        if not isinstance(key, StateKey):
            return False
        if int(getattr(key, "k", 0)) != 0:
            return False
        if getattr(key, "dtype", None) != self.dtype:
            return False
        owner = getattr(key, "owner", None)
        if getattr(owner, "owner_type", None) != self.owner_type:
            return False
        owner_name = getattr(owner, "owner_name", None)
        if not isinstance(owner_name, str) or owner_name == "":
            return False
        return True

    def _canonical_value_key(self, key: StateKey) -> StateKey:
        return StateKey(
            k=int(key.k),
            owner=key.owner,
            dtype=self.dtype,
            field=self.field,
            frame=getattr(key, "frame", None),
            rel_frame=getattr(key, "rel_frame", None),
        )

    def _state_ref(self, key: StateKey) -> Any:
        if key in self._state_ref_cache:
            return self._state_ref_cache[key]
        state_ref = self._resolve_state_ref(key)
        self._state_ref_cache[key] = state_ref
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

        x_all = np.asarray(x_all, dtype=float).reshape(-1)

        q = x_all
        if pack is not None and hasattr(pack, "slices") and self.q_var in getattr(pack, "slices", {}):
            s, e = pack.slices[self.q_var]
            q = np.asarray(x_all[s:e], dtype=float).reshape(-1)

        self._update_kinematics(q)

        value_field = self.field
        jac_value_field = jac_field(value_field, var=self.q_var)

        needs: dict[StateKey, tuple[bool, bool]] = {}
        for key in required:
            if not self._accept_required_key(key):
                continue

            value_key = self._canonical_value_key(key)
            need_val, need_jac = needs.get(value_key, (False, False))
            if key.field == value_field:
                need_val = True
            elif key.field == jac_value_field:
                need_jac = True
            else:
                continue
            needs[value_key] = (need_val, need_jac)

        if not needs:
            return {}

        value_by_key: dict[StateKey, Array] = {}
        jac_by_key: dict[StateKey, Array] = {}

        for value_key, (need_val, need_jac) in needs.items():
            state_ref = self._state_ref(value_key)

            if need_val:
                value = np.asarray(self._state_value(q, value_key, state_ref), dtype=float).reshape(-1)
                value_by_key[value_key] = value.copy()

            if need_jac:
                jac = np.asarray(self._state_jacobian(q, value_key, state_ref), dtype=float)
                jac_by_key[value_key] = jac.copy()

        out: dict[StateKey, Any] = {}
        for key in required:
            if not self._accept_required_key(key):
                continue
            value_key = self._canonical_value_key(key)
            if value_key not in needs:
                continue

            if key.field == value_field and value_key in value_by_key:
                out[key] = value_by_key[value_key]
            elif key.field == jac_value_field and value_key in jac_by_key:
                out[key] = jac_by_key[value_key]

        return out
