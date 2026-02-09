from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Iterable
import numpy as np

Array = np.ndarray


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
                raise KeyError(f"StateCache.build_state missing keys: {missing_after}")
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
