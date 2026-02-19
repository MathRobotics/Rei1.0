from __future__ import annotations

import pytest

from dataclasses import dataclass
from typing import Any

import numpy as np

from eiopt.backends.state.dispatch.composite import CompositeStateBuilder
from eiopt.backends.state.dispatch.template import BackendDispatchStateBuilder
from eiopt.core.state_cache import OwnerKey, StateKey

def _key(*, k: int, dtype: str, owner_type: str, owner_name: str, field: str) -> StateKey:
    return StateKey(
        k=int(k),
        owner=OwnerKey(owner_type=owner_type, owner_name=owner_name),
        dtype=str(dtype),
        field=str(field),
    )

@dataclass
class _ConstProvider:
    dtype: str
    owner_type: str
    field: str
    value: float
    drop_last: bool = False

    def accepts(self, key: StateKey) -> bool:
        owner = getattr(key, "owner", None)
        return (
            str(getattr(key, "dtype", "")) == self.dtype
            and str(getattr(owner, "owner_type", "")) == self.owner_type
            and str(getattr(key, "field", "")) == self.field
        )

    def build_state(
        self,
        x_all: np.ndarray,
        *,
        pack: Any = None,
        time: Any = None,
        required: list[StateKey] | tuple[StateKey, ...] | None = None,
    ) -> dict[StateKey, Any]:
        del x_all, pack, time
        if required is None:
            return {}
        keys = list(required)
        if self.drop_last and len(keys) > 0:
            keys = keys[:-1]
        return {key: np.array([self.value], dtype=float) for key in keys}

class _DummyDispatchBuilder(BackendDispatchStateBuilder):
    def __init__(self, *, allow_nonzero_k: bool = False) -> None:
        super().__init__(
            model=object(),
            data={},
            allow_nonzero_k=allow_nonzero_k,
        )
        self.register_value_and_jac(
            dtype="vision",
            owner_type="camera",
            field="proj",
            value_handler=self._handle_value,
            jac_handler=self._handle_jac,
        )

    def _resolve_state_ref(self, key: StateKey) -> Any:
        return key

    @staticmethod
    def _handle_value(q: np.ndarray, key: StateKey, state_ref: Any) -> np.ndarray:
        del q, key, state_ref
        return np.array([1.0], dtype=float)

    @staticmethod
    def _handle_jac(q: np.ndarray, key: StateKey, state_ref: Any) -> np.ndarray:
        del key, state_ref
        n = int(np.asarray(q, dtype=float).reshape(-1).size)
        return np.zeros((1, n), dtype=float)

class TestCompositeStateBuilder:
    @staticmethod
    def _build(builder: CompositeStateBuilder, *required: StateKey) -> dict[StateKey, Any]:
        return builder.build_state(
            np.array([0.0], dtype=float),
            required=list(required),
        )

    def test_composite_state_builder_dispatches_required_keys(self) -> None:
        key_kin = _key(k=0, dtype="kinematics", owner_type="link", owner_name="ee", field="pos")
        key_vis = _key(k=2, dtype="vision", owner_type="camera", owner_name="cam0", field="proj")
        p_kin = _ConstProvider(dtype="kinematics", owner_type="link", field="pos", value=1.0)
        p_vis = _ConstProvider(dtype="vision", owner_type="camera", field="proj", value=2.0)
        builder = CompositeStateBuilder([p_kin, p_vis])

        out = self._build(builder, key_kin, key_vis)
        assert set(out.keys()) == {key_kin, key_vis}
        assert np.allclose(out[key_kin], np.array([1.0], dtype=float))
        assert np.allclose(out[key_vis], np.array([2.0], dtype=float))

    def test_composite_state_builder_rejects_ambiguous_provider_match(self) -> None:
        key_vis = _key(k=0, dtype="vision", owner_type="camera", owner_name="cam0", field="proj")
        p1 = _ConstProvider(dtype="vision", owner_type="camera", field="proj", value=1.0)
        p2 = _ConstProvider(dtype="vision", owner_type="camera", field="proj", value=2.0)
        builder = CompositeStateBuilder([p1, p2])

        with pytest.raises(ValueError):
            _ = self._build(builder, key_vis)

    def test_composite_state_builder_detects_missing_required_keys(self) -> None:
        key_vis = _key(k=0, dtype="vision", owner_type="camera", owner_name="cam0", field="proj")
        p = _ConstProvider(dtype="vision", owner_type="camera", field="proj", value=2.0, drop_last=True)
        builder = CompositeStateBuilder([p])

        with pytest.raises(KeyError):
            _ = self._build(builder, key_vis)

    def test_composite_state_builder_can_ignore_unmatched_keys(self) -> None:
        key_kin = _key(k=0, dtype="kinematics", owner_type="link", owner_name="ee", field="pos")
        key_unknown = _key(k=0, dtype="unknown", owner_type="x", owner_name="y", field="z")
        p_kin = _ConstProvider(dtype="kinematics", owner_type="link", field="pos", value=1.0)
        builder = CompositeStateBuilder([p_kin], allow_unmatched_keys=True)

        out = self._build(builder, key_kin, key_unknown)
        assert set(out.keys()) == {key_kin}

class TestBackendDispatchStateBuilder:
    def test_backend_dispatch_builder_accepts_can_enable_nonzero_k(self) -> None:
        key_k0 = _key(k=0, dtype="vision", owner_type="camera", owner_name="cam0", field="proj")
        key_k2 = _key(k=2, dtype="vision", owner_type="camera", owner_name="cam0", field="proj")

        default_builder = _DummyDispatchBuilder(allow_nonzero_k=False)
        assert default_builder.accepts(key_k0)
        assert not (default_builder.accepts(key_k2))

        extended_builder = _DummyDispatchBuilder(allow_nonzero_k=True)
        assert extended_builder.accepts(key_k0)
        assert extended_builder.accepts(key_k2)

