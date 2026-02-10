from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import numpy as np

from ..core.state_cache import StateKey
from ..core.state_schema import DTYPE_FRAME, jac_field

Array = np.ndarray


class BackendFramePosStateBuilder:
    """Template backend -> build_state() bridge for `dtype="frame", field="pos"` keys.

    Copy this file when adding a new backend, or subclass it.

    Expected keys (k=0 only):
      - dtype="frame", owner_type="link", field="pos"
      - dtype="frame", owner_type="link", field="pos_J_<q_var>"

    Notes:
      - Keep this module importable without heavy dependencies.
        If you must import a backend package, wrap it in try/except.
      - `build_state()` should return only the keys requested in `required`.
    """

    def __init__(self, model: Any, data: Any, *, q_var: str = "q") -> None:
        self.model = model
        self.data = data
        self.q_var = str(q_var)
        self._frame_ref_cache: dict[str, Any] = {}
        # Backward-compatible alias (legacy name used by old subclasses/tests).
        self._frame_id_cache = self._frame_ref_cache

    def _update_kinematics(self, q: Array) -> None:
        """Run backend FK/Jacobian prerequisites for the current `q`."""

        raise NotImplementedError("TODO: implement backend kinematics update.")

    def _resolve_frame_ref(self, frame_name: str) -> Any:
        """Resolve backend-specific frame reference from a frame name.

        The reference may be an integer frame id, a string name, or any
        backend-native handle.
        """

        # Backward-compatible fallback for older subclasses.
        return self._resolve_frame_id(frame_name)

    def _resolve_frame_id(self, frame_name: str) -> int:
        """Legacy API: resolve backend-specific frame id from a frame name."""

        raise NotImplementedError(
            "TODO: implement backend frame name resolution. "
            "Override `_resolve_frame_ref()` (preferred) or `_resolve_frame_id()`."
        )

    def _frame_pos(self, frame_ref: Any) -> Array:
        """Return frame position (3,) for the given backend `frame_ref`."""

        raise NotImplementedError("TODO: implement backend position extraction.")

    def _frame_pos_jacobian(self, q: Array, frame_ref: Any) -> Array:
        """Return linear position Jacobian (3,n) for backend `frame_ref`.

        If your backend provides a 6D spatial Jacobian, the (linear, angular) row
        order is library-dependent. Consider using helpers in `eiopt.backends._spatial`
        and define the convention in your backend wrapper.
        """

        raise NotImplementedError("TODO: implement backend Jacobian extraction.")

    def _frame_ref(self, frame_name: str) -> Any:
        name = str(frame_name)
        if name in self._frame_ref_cache:
            return self._frame_ref_cache[name]
        frame_ref = self._resolve_frame_ref(name)
        self._frame_ref_cache[name] = frame_ref
        return frame_ref

    def _frame_id(self, frame_name: str) -> Any:
        """Legacy alias for `_frame_ref()`."""

        return self._frame_ref(frame_name)

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
            frame_ref = self._frame_ref(frame_name)

            if need_pos:
                pos = np.asarray(self._frame_pos(frame_ref), dtype=float).reshape(3)
                pos_by_frame[frame_name] = pos.copy()

            if need_jac:
                Jpos = np.asarray(self._frame_pos_jacobian(q, frame_ref), dtype=float)
                Jpos_by_frame[frame_name] = Jpos.copy()

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
