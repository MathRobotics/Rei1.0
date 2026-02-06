from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Iterable

import inspect
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

    `build_state` should ideally accept:
      build_state(x_all, pack=pack, time=time_grid, required=required_keys) -> dict[StateKey, Any]

    Backward compatible:
      build_state(x_all) -> dict
      build_state(x_all, time=time_grid) -> dict
    """

    build_state: Callable[..., dict]

    state: dict[StateKey, Any] = field(default_factory=dict)

    _rev_last: int = -1
    _time_rev_last: int = -1
    _req_sig_last: int = 0

    _memo: dict[StateKey, Any] = field(default_factory=dict)

    _build_state_accepts: set[str] | None = field(default=None, init=False, repr=False)
    _build_state_accepts_kwargs: bool = field(default=False, init=False, repr=False)

    def invalidate(self) -> None:
        self._rev_last = -1
        self._time_rev_last = -1
        self._req_sig_last = 0
        self._memo.clear()
        self.state.clear()

    def _required_sig(self, required: Optional[Iterable[StateKey]]) -> int:
        if required is None:
            return 0
        return hash(frozenset(required))

    def _analyze_build_state(self) -> None:
        if self._build_state_accepts is not None:
            return
        try:
            sig = inspect.signature(self.build_state)
        except (TypeError, ValueError):
            self._build_state_accepts = set()
            self._build_state_accepts_kwargs = True
            return

        self._build_state_accepts = set(sig.parameters.keys())
        self._build_state_accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

    def update_if_needed(
        self,
        pack: PackLike,
        *,
        time: Any = None,
        required: Optional[Iterable[StateKey]] = None,
    ) -> None:
        rev = int(getattr(pack, "revision", 0))
        time_rev = int(getattr(time, "revision", 0)) if time is not None else 0
        req_sig = self._required_sig(required)

        if rev == self._rev_last and time_rev == self._time_rev_last and req_sig == self._req_sig_last:
            return

        x_all = np.asarray(pack.get(), dtype=float).reshape(-1)

        self._analyze_build_state()
        accepts = self._build_state_accepts or set()
        accepts_kwargs = bool(self._build_state_accepts_kwargs)

        kwargs: dict[str, Any] = {}
        if accepts_kwargs or "pack" in accepts:
            kwargs["pack"] = pack
        if accepts_kwargs or "time" in accepts:
            kwargs["time"] = time
        if accepts_kwargs or "required" in accepts:
            kwargs["required"] = required

        st = self.build_state(x_all, **kwargs) if kwargs else self.build_state(x_all)

        if not isinstance(st, dict):
            raise TypeError("StateCache.build_state must return a dict.")

        self.state = st
        self._rev_last = rev
        self._time_rev_last = time_rev
        self._req_sig_last = req_sig
        self._memo.clear()

    def get(self, key: StateKey) -> Any:
        if key in self._memo:
            return self._memo[key]
        if key not in self.state:
            raise KeyError(f"StateCache: missing key: {key}")
        return self.state[key]

    def set_memo(self, key: StateKey, value: Any) -> None:
        self._memo[key] = value
