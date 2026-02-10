from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import numpy as np

from ..core.state_cache import StateKey
from ..core.state_schema import DTYPE_KINEMATICS, jac_field

Array = np.ndarray


class BackendFramePosStateBuilder:
    """Template backend -> build_state() bridge for `dtype="kinematics", field="pos"` keys.

    Copy this file when adding a new backend, or subclass it.

    Expected keys (k=0 only):
      - dtype="kinematics", owner_type="link", field="pos"
      - dtype="kinematics", owner_type="link", field="pos_J_<q_var>"

    Notes:
      - Keep this module importable without heavy dependencies.
        If you must import a backend package, wrap it in try/except.
      - `build_state()` should return only the keys requested in `required`.
    """

    def __init__(self, model: Any, data: Any, *, q_var: str = "q") -> None:
        self.model = model
        self.data = data
        self.q_var = str(q_var)
        self._state_ref_cache: dict[StateKey, Any] = {}

    def _update_kinematics(self, q: Array) -> None:
        """Run backend FK/Jacobian prerequisites for the current `q`."""

        raise NotImplementedError("TODO: implement backend kinematics update.")

    def _resolve_state_ref(self, key: StateKey) -> Any:
        """Resolve backend-specific state reference from a requested StateKey."""

        raise NotImplementedError("TODO: implement backend state reference resolution.")

    def _frame_pos(self, state_ref: Any) -> Array:
        """Return frame position (3,) for the given backend `state_ref`."""

        raise NotImplementedError("TODO: implement backend position extraction.")

    def _frame_pos_jacobian(self, q: Array, state_ref: Any) -> Array:
        """Return linear position Jacobian (3,n) for backend `state_ref`.

        If your backend provides a 6D spatial Jacobian, the (linear, angular) row
        order is library-dependent. Consider using helpers in `eiopt.backends._spatial`
        and define the convention in your backend wrapper.
        """

        raise NotImplementedError("TODO: implement backend Jacobian extraction.")

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

        pos_field = "pos"
        jac_pos_field = jac_field(pos_field, var=self.q_var)

        needs: dict[StateKey, tuple[bool, bool]] = {}
        for key in required:
            if not isinstance(key, StateKey):
                continue
            if int(getattr(key, "k", 0)) != 0:
                continue
            if getattr(key, "dtype", None) != DTYPE_KINEMATICS:
                continue
            owner = getattr(key, "owner", None)
            if getattr(owner, "owner_type", None) != "link":
                continue
            frame_name = getattr(owner, "owner_name", None)
            if not isinstance(frame_name, str) or frame_name == "":
                continue

            pos_key = StateKey(
                k=int(key.k),
                owner=owner,
                dtype=str(key.dtype),
                field=pos_field,
                frame=getattr(key, "frame", None),
                rel_frame=getattr(key, "rel_frame", None),
            )
            need_pos, need_jac = needs.get(pos_key, (False, False))
            if key.field == pos_field:
                need_pos = True
            elif key.field == jac_pos_field:
                need_jac = True
            else:
                continue
            needs[pos_key] = (need_pos, need_jac)

        if not needs:
            return {}

        pos_by_key: dict[StateKey, Array] = {}
        Jpos_by_key: dict[StateKey, Array] = {}

        for pos_key, (need_pos, need_jac) in needs.items():
            state_ref = self._state_ref(pos_key)

            if need_pos:
                pos = np.asarray(self._frame_pos(state_ref), dtype=float).reshape(3)
                pos_by_key[pos_key] = pos.copy()

            if need_jac:
                Jpos = np.asarray(self._frame_pos_jacobian(q, state_ref), dtype=float)
                Jpos_by_key[pos_key] = Jpos.copy()

        out: dict[StateKey, Any] = {}
        for key in required:
            if not isinstance(key, StateKey):
                continue
            if int(getattr(key, "k", 0)) != 0:
                continue
            if getattr(key, "dtype", None) != DTYPE_KINEMATICS:
                continue
            owner = getattr(key, "owner", None)
            if getattr(owner, "owner_type", None) != "link":
                continue
            frame_name = getattr(owner, "owner_name", None)
            if not isinstance(frame_name, str) or frame_name == "":
                continue

            pos_key = StateKey(
                k=int(key.k),
                owner=owner,
                dtype=str(key.dtype),
                field=pos_field,
                frame=getattr(key, "frame", None),
                rel_frame=getattr(key, "rel_frame", None),
            )
            if pos_key not in needs:
                continue

            if key.field == pos_field and pos_key in pos_by_key:
                out[key] = pos_by_key[pos_key]
            elif key.field == jac_pos_field and pos_key in Jpos_by_key:
                out[key] = Jpos_by_key[pos_key]

        return out
