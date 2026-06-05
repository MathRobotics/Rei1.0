from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ....core.state_cache import StateKey

Array = np.ndarray


def _state_key_label(key: StateKey | None) -> str:
    if key is None:
        return "<unknown>"
    owner = getattr(key, "owner", None)
    owner_type = getattr(owner, "owner_type", None)
    owner_name = getattr(owner, "owner_name", None)
    return (
        f"k={getattr(key, 'k', None)!r}, "
        f"dtype={getattr(key, 'dtype', None)!r}, "
        f"owner_type={owner_type!r}, "
        f"owner_name={owner_name!r}, "
        f"field={getattr(key, 'field', None)!r}"
    )


def _assert_numeric_array(
    value: Any,
    *,
    key: StateKey,
    expected_shape: tuple[int, ...] | None,
    helper_name: str,
) -> Array:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"{helper_name}: provider returned non-numeric data for key=({_state_key_label(key)})."
        ) from exc

    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        raise AssertionError(
            f"{helper_name}: shape mismatch for key=({_state_key_label(key)}). "
            f"Expected shape={tuple(expected_shape)}, actual shape={tuple(arr.shape)}."
        )
    return arr


def assert_provider_contract(
    provider: Any,
    sample_q: Any,
    expected_fields: Sequence[StateKey],
    *,
    expected_shapes: Mapping[StateKey, tuple[int, ...]] | None = None,
) -> dict[StateKey, Array]:
    """Assert that a single-step robotics provider satisfies Rei's state contract."""

    keys = list(expected_fields)
    if len(keys) == 0:
        raise AssertionError("assert_provider_contract: expected_fields must be non-empty.")
    if not hasattr(provider, "accepts") or not hasattr(provider, "build_state"):
        raise AssertionError("assert_provider_contract: provider must expose accepts() and build_state().")

    unsupported = [key for key in keys if not bool(provider.accepts(key))]
    if unsupported:
        labels = ", ".join(_state_key_label(key) for key in unsupported)
        raise AssertionError(f"assert_provider_contract: provider does not accept expected keys: {labels}.")

    out = provider.build_state(np.asarray(sample_q, dtype=float).reshape(-1), required=keys)
    missing = [key for key in keys if key not in out]
    if missing:
        labels = ", ".join(_state_key_label(key) for key in missing)
        raise AssertionError(f"assert_provider_contract: build_state() did not return expected keys: {labels}.")

    shapes = {} if expected_shapes is None else dict(expected_shapes)
    checked: dict[StateKey, Array] = {}
    for key in keys:
        checked[key] = _assert_numeric_array(
            out[key],
            key=key,
            expected_shape=shapes.get(key),
            helper_name="assert_provider_contract",
        )
    return checked


def assert_trajectory_provider_contract(
    provider: Any,
    sample_p: Any,
    expected_keys: Sequence[StateKey],
    *,
    expected_shapes: Mapping[StateKey, tuple[int, ...]] | None = None,
) -> dict[StateKey, Array]:
    """Assert that a trajectory robotics provider satisfies Rei's state contract."""

    keys = list(expected_keys)
    if len(keys) == 0:
        raise AssertionError("assert_trajectory_provider_contract: expected_keys must be non-empty.")
    try:
        p_vec = np.asarray(sample_p, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise AssertionError("assert_trajectory_provider_contract: sample_p must be numeric.") from exc
    return assert_provider_contract(
        provider,
        p_vec,
        keys,
        expected_shapes=expected_shapes,
    )


__all__ = [
    "assert_provider_contract",
    "assert_trajectory_provider_contract",
]
