from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Iterable
import numpy as np

Array = np.ndarray


def _format_missing_state_keys(keys: Iterable["StateKey"], *, max_groups: int = 8) -> str:
    grouped: dict[tuple[str, str, str, str, str | None, str | None], set[int]] = defaultdict(set)
    for key in keys:
        owner = getattr(key, "owner", None)
        grouped[
            (
                str(getattr(key, "dtype", "")),
                str(getattr(owner, "owner_type", "")),
                str(getattr(owner, "owner_name", "")),
                str(getattr(key, "field", "")),
                getattr(key, "frame", None),
                getattr(key, "rel_frame", None),
            )
        ].add(int(getattr(key, "k", 0)))

    items = sorted(grouped.items(), key=lambda item: item[0])
    lines: list[str] = []
    for idx, (sig, ks) in enumerate(items):
        if idx >= int(max_groups):
            lines.append(f"... and {len(items) - max_groups} more group(s)")
            break
        dtype, owner_type, owner_name, field, frame, rel_frame = sig
        ks_sorted = sorted(ks)
        frame_info = []
        if frame is not None:
            frame_info.append(f"frame={frame!r}")
        if rel_frame is not None:
            frame_info.append(f"rel_frame={rel_frame!r}")
        frame_suffix = "" if len(frame_info) == 0 else f" ({', '.join(frame_info)})"
        lines.append(
            f"- dtype={dtype!r}, owner={owner_type!r}:{owner_name!r}, field={field!r}, ks={ks_sorted}{frame_suffix}"
        )
    return "\n".join(lines)


class PackLike(Protocol):
    revision: int

    def get(self) -> Array: ...


@dataclass(frozen=True)
class OwnerKey:
    owner_type: str
    owner_name: str


@dataclass(frozen=True)
class StateKey:
    k: int
    owner: OwnerKey
    dtype: str
    field: str
    frame: Optional[str] = None
    rel_frame: Optional[str] = None


@dataclass
class StateCache:
    """Cache for expensive state computations.

    `build_state` must accept:
      build_state(x_all, pack=pack, time=time_grid, required=required_keys) -> dict[StateKey, Any]

    Required handling:
      - If `required` is provided, StateCache unions requests and only recomputes
        missing keys (for the same variable/time revision).
      - If `required` is None, it is treated as "compute all" and cached as complete.
    """

    build_state: Callable[..., dict]

    state: dict[StateKey, Any] = field(default_factory=dict)

    _rev_last: int = -1
    _time_rev_last: int = -1
    _required_cached: set[StateKey] = field(default_factory=set)
    _all_valid: bool = False

    _memo: dict[StateKey, Any] = field(default_factory=dict)
    _pack_last: PackLike | None = None
    _time_last: Any = None

    def invalidate(self) -> None:
        self._rev_last = -1
        self._time_rev_last = -1
        self._required_cached.clear()
        self._all_valid = False
        self._memo.clear()
        self.state.clear()

    def update_if_needed(
        self,
        pack: PackLike,
        *,
        time: Any = None,
        required: Optional[Iterable[StateKey]] = None,
    ) -> None:
        rev = int(getattr(pack, "revision", 0))
        time_rev = int(getattr(time, "revision", 0)) if time is not None else 0

        if rev != self._rev_last or time_rev != self._time_rev_last:
            self._rev_last = rev
            self._time_rev_last = time_rev
            self._required_cached.clear()
            self._all_valid = False
            self._memo.clear()
            self.state.clear()

        missing: set[StateKey] | None = None

        if required is None:
            if self._all_valid:
                return
        else:
            if self._all_valid:
                return
            req_set = set(required)
            missing = req_set - self._required_cached
            if not missing:
                return

        x_all = np.asarray(pack.get(), dtype=float).reshape(-1)
        self._pack_last = pack
        self._time_last = time
        st = self.build_state(x_all, pack=pack, time=time, required=missing if required is not None else None)

        if not isinstance(st, dict):
            raise TypeError("StateCache.build_state must return a dict.")

        if required is None:
            self.state = st
            self._required_cached = set(st.keys())
            self._all_valid = True
        else:
            missing = missing or set()
            missing_after = missing - set(st.keys())
            if missing_after:
                summary = _format_missing_state_keys(missing_after)
                raise KeyError(
                    "StateCache.build_state missing required keys.\n"
                    f"{summary}\n"
                    "Ensure build_state returns every requested StateKey for the selected backend/settings."
                )
            if not self.state:
                self.state = {}
            self.state.update(st)
            self._required_cached |= set(st.keys())

        self._memo.clear()

    def get(self, key: StateKey) -> Any:
        if key in self._memo:
            return self._memo[key]
        if key not in self.state:
            raise KeyError(f"StateCache: missing key: {key}")
        return self.state[key]

    def set_memo(self, key: StateKey, value: Any) -> None:
        self._memo[key] = value

    def jacobian_transpose_mul(self, value_key: StateKey, jac_key: StateKey, rhs: Array | Any) -> Array:
        """Compute a state Jacobian VJP through the backend when it exposes one."""

        _base_field, sep, jac_var = str(jac_key.field).partition("_J_")
        if not sep or _base_field == "" or jac_var == "":
            raise AttributeError("StateCache: jac_key is not a Jacobian StateKey.")

        builder = getattr(self.build_state, "__self__", None)
        if builder is None:
            raise AttributeError("StateCache: build_state is not bound to a backend builder.")

        pack = self._pack_last
        if pack is None:
            raise AttributeError("StateCache: no active VariablePack for backend VJP.")
        x_all = np.asarray(pack.get(), dtype=float).reshape(-1)

        p_var = getattr(builder, "p_var", None)
        param_vjp = getattr(builder, "param_jacobian_transpose_mul", None)
        if callable(param_vjp) and isinstance(p_var, str) and jac_var == p_var:
            return np.asarray(
                param_vjp(
                    x_all,
                    value_key,
                    rhs,
                    pack=pack,
                    time=self._time_last,
                    update_kinematics=False,
                ),
                dtype=float,
            )

        q_var = getattr(builder, "q_var", None)
        state_vjp = getattr(builder, "jacobian_transpose_mul", None)
        if callable(state_vjp) and isinstance(q_var, str) and jac_var == q_var:
            return np.asarray(
                state_vjp(
                    x_all,
                    value_key,
                    rhs,
                    update_kinematics=False,
                ),
                dtype=float,
            )

        raise AttributeError(
            "StateCache: backend does not expose a matching jacobian_transpose_mul "
            f"for var {jac_var!r}."
        )
