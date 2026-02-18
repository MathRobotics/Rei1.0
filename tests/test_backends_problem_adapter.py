from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np

from eiopt.optimize_backends.problem_adapter import compile_problem_with_adapter


def _basic_dsl() -> dict[str, Any]:
    return {
        "variables": [{"name": "x", "dim": 1, "init": [2.0]}],
        "terms": [
            {
                "expr": {"type": "get_var", "name": "x_identity", "var": "x"},
                "cost": {"type": "l2"},
            }
        ],
    }


@dataclass
class _NoopStateBuilder:
    def build_state(self, x_all: np.ndarray, *, pack: Any = None, time: Any = None, required: Any = None) -> dict:
        del x_all, pack, time, required
        return {}


@dataclass(frozen=True)
class _PreparedDsl:
    dsl: dict[str, Any]
    name: str


@dataclass
class _FakeAdapter:
    use_wrapped_prepared: bool = True
    validated: bool = False
    seen_prepared: Any = None

    def prepare_dsl(self, dsl: dict[str, Any]) -> Any:
        if self.use_wrapped_prepared:
            return _PreparedDsl(dsl=dict(dsl), name="prepared")
        return dict(dsl)

    def build_state_builder(self, *, model: Any, data: Any, prepared: Any) -> _NoopStateBuilder:
        del model, data
        self.seen_prepared = prepared
        return _NoopStateBuilder()

    def validate_runtime(self, *, runtime: Any, state_builder: Any, prepared: Any) -> None:
        self.validated = True
        self.seen_prepared = prepared
        if not isinstance(state_builder, _NoopStateBuilder):
            raise AssertionError("validate_runtime: unexpected state_builder.")
        if int(runtime.pack.n_total) != 1:
            raise AssertionError("validate_runtime: unexpected variable dimension.")


@dataclass
class _BadAdapter(_FakeAdapter):
    def build_state_builder(self, *, model: Any, data: Any, prepared: Any) -> Any:
        del model, data, prepared
        return object()


class TestBackendsProblemAdapter(unittest.TestCase):
    def test_compile_problem_with_adapter_accepts_wrapped_prepared_dsl(self) -> None:
        adapter = _FakeAdapter(use_wrapped_prepared=True)
        compiled = compile_problem_with_adapter(
            _basic_dsl(),
            model=object(),
            data={},
            adapter=adapter,
        )

        self.assertTrue(adapter.validated)
        self.assertEqual(getattr(compiled.prepared, "name", None), "prepared")
        r, J = compiled.runtime.linearize()
        self.assertTrue(np.allclose(r, np.array([2.0], dtype=float)))
        self.assertTrue(np.allclose(J, np.array([[1.0]], dtype=float)))

    def test_compile_problem_with_adapter_accepts_mapping_prepared_dsl(self) -> None:
        adapter = _FakeAdapter(use_wrapped_prepared=False)
        compiled = compile_problem_with_adapter(
            _basic_dsl(),
            model=object(),
            data={},
            adapter=adapter,
        )

        self.assertTrue(adapter.validated)
        self.assertIsInstance(compiled.prepared, dict)

    def test_compile_problem_with_adapter_requires_build_state(self) -> None:
        adapter = _BadAdapter(use_wrapped_prepared=True)
        with self.assertRaises(TypeError):
            _ = compile_problem_with_adapter(
                _basic_dsl(),
                model=object(),
                data={},
                adapter=adapter,
            )


if __name__ == "__main__":
    unittest.main()
