from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ...core.state_cache import StateKey

Array = np.ndarray


@runtime_checkable
class StateProvider(Protocol):
    """Minimal protocol for pluggable state providers."""

    def accepts(self, key: StateKey) -> bool: ...

    def build_state(
        self,
        x_all: Array,
        *,
        pack: Any = None,
        time: Any = None,
        required: Iterable[StateKey] | None = None,
    ) -> dict[StateKey, Any]: ...


def _provider_name(provider: Any) -> str:
    return f"{provider.__class__.__module__}.{provider.__class__.__qualname__}"


def _format_key(key: StateKey) -> str:
    owner = getattr(key, "owner", None)
    return (
        "StateKey("
        f"k={int(getattr(key, 'k', 0))}, "
        f"dtype={getattr(key, 'dtype', None)!r}, "
        f"owner_type={getattr(owner, 'owner_type', None)!r}, "
        f"owner_name={getattr(owner, 'owner_name', None)!r}, "
        f"field={getattr(key, 'field', None)!r}, "
        f"frame={getattr(key, 'frame', None)!r}, "
        f"rel_frame={getattr(key, 'rel_frame', None)!r})"
    )


class CompositeStateBuilder:
    """Compose multiple `StateProvider` objects behind one `build_state()`."""

    def __init__(
        self,
        providers: Sequence[StateProvider],
        *,
        allow_unmatched_keys: bool = False,
    ) -> None:
        if len(providers) == 0:
            raise ValueError("CompositeStateBuilder: providers must be non-empty.")
        self.providers: tuple[StateProvider, ...] = tuple(providers)
        self.allow_unmatched_keys = bool(allow_unmatched_keys)
        self._provider_index_cache: dict[StateKey, int] = {}

        for idx, provider in enumerate(self.providers):
            accepts = getattr(provider, "accepts", None)
            build_state = getattr(provider, "build_state", None)
            if not callable(accepts):
                raise TypeError(
                    "CompositeStateBuilder: each provider must define callable accepts(key). "
                    f"providers[{idx}]={_provider_name(provider)!r}"
                )
            if not callable(build_state):
                raise TypeError(
                    "CompositeStateBuilder: each provider must define callable build_state(...). "
                    f"providers[{idx}]={_provider_name(provider)!r}"
                )

    def _resolve_provider_index(self, key: StateKey) -> int | None:
        cached = self._provider_index_cache.get(key, None)
        if cached is not None:
            provider = self.providers[cached]
            if bool(provider.accepts(key)):
                return int(cached)
            del self._provider_index_cache[key]

        matches = [i for i, provider in enumerate(self.providers) if bool(provider.accepts(key))]
        if len(matches) == 1:
            idx = int(matches[0])
            self._provider_index_cache[key] = idx
            return idx
        if len(matches) == 0:
            if self.allow_unmatched_keys:
                return None
            raise KeyError(
                "CompositeStateBuilder: no provider accepts key: "
                f"{_format_key(key)}"
            )

        providers_str = ", ".join(_provider_name(self.providers[i]) for i in matches)
        raise ValueError(
            "CompositeStateBuilder: multiple providers accept the same key. "
            f"key={_format_key(key)} matches=[{providers_str}]"
        )

    def _merge_provider_state(
        self,
        merged: dict[StateKey, Any],
        provider_state: dict[StateKey, Any],
        *,
        provider_name: str,
    ) -> None:
        for key, value in provider_state.items():
            if key in merged:
                raise ValueError(
                    "CompositeStateBuilder: duplicate key produced by providers. "
                    f"provider={provider_name!r}, key={_format_key(key)}"
                )
            merged[key] = value

    @staticmethod
    def _coerce_required(required: Iterable[StateKey] | None) -> tuple[StateKey, ...] | None:
        if required is None:
            return None
        seen: set[StateKey] = set()
        out: list[StateKey] = []
        for key in required:
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return tuple(out)

    def build_state(
        self,
        x_all: Array,
        *,
        pack: Any = None,
        time: Any = None,
        required: Iterable[StateKey] | None = None,
    ) -> dict[StateKey, Any]:
        required_keys = self._coerce_required(required)

        if required_keys is None:
            merged: dict[StateKey, Any] = {}
            for provider in self.providers:
                provider_name = _provider_name(provider)
                st = provider.build_state(x_all, pack=pack, time=time, required=None)
                if not isinstance(st, dict):
                    raise TypeError(
                        "CompositeStateBuilder: provider build_state must return dict. "
                        f"provider={provider_name!r}, got={type(st).__name__!r}"
                    )
                self._merge_provider_state(merged, st, provider_name=provider_name)
            return merged

        if len(required_keys) == 0:
            return {}

        grouped_keys: dict[int, list[StateKey]] = {}
        for key in required_keys:
            provider_idx = self._resolve_provider_index(key)
            if provider_idx is None:
                continue
            grouped_keys.setdefault(int(provider_idx), []).append(key)

        merged: dict[StateKey, Any] = {}
        for provider_idx, keys in grouped_keys.items():
            provider = self.providers[provider_idx]
            provider_name = _provider_name(provider)
            st = provider.build_state(x_all, pack=pack, time=time, required=keys)
            if not isinstance(st, dict):
                raise TypeError(
                    "CompositeStateBuilder: provider build_state must return dict. "
                    f"provider={provider_name!r}, got={type(st).__name__!r}"
                )
            missing = [key for key in keys if key not in st]
            if missing:
                miss = ", ".join(_format_key(key) for key in missing[:6])
                more = "" if len(missing) <= 6 else f", ... (+{len(missing) - 6})"
                raise KeyError(
                    "CompositeStateBuilder: provider did not return required keys. "
                    f"provider={provider_name!r}, missing=[{miss}{more}]"
                )

            selected = {key: st[key] for key in keys}
            self._merge_provider_state(merged, selected, provider_name=provider_name)

        return merged


__all__ = [
    "StateProvider",
    "CompositeStateBuilder",
]
