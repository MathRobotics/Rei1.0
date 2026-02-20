from __future__ import annotations

import pytest

from dataclasses import dataclass
from typing import Any

import numpy as np

from rei.optimize_backends.trajectory_adapter import compile_trajectory_problem_with_adapter

class _NoopStateBuilder:
    def build_state(self, q: np.ndarray, key: Any) -> dict[str, Any]:
        del q, key
        return {}

@dataclass
class _FakeAdapter:
    dof: int = 1
    order: int = 3
    validated: bool = False
    seen_p_var: str | None = None
    builder: _NoopStateBuilder | None = None

    def infer_model_dof(self, model: Any) -> int | None:
        del model
        return int(self.dof)

    def infer_model_order(self, model: Any) -> int:
        del model
        return int(self.order)

    def build_state_builder(self, *, model: Any, data: Any, prepared: Any) -> _NoopStateBuilder:
        del model, data
        self.seen_p_var = str(prepared.p_var)
        self.builder = _NoopStateBuilder()
        return self.builder

    def validate_runtime(self, *, runtime: Any, state_builder: Any, prepared: Any) -> None:
        self.validated = True
        if state_builder is not self.builder:
            raise AssertionError("validate_runtime: state_builder identity mismatch.")
        if int(runtime.pack.n_total) != int(prepared.trajectory_map.p_dim):
            raise AssertionError("validate_runtime: runtime pack size mismatch.")

class TestBackendsTrajectoryAdapter:
    def _linear_dsl(self, *, include_q_dim: bool = True) -> dict:
        trajectory = {
            "type": "linear",
            "var": "p",
            "steps": 2,
            "A": [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        }
        if include_q_dim:
            trajectory["q_dim"] = 1
        return {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": trajectory,
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "name": "p_minus_ref",
                        "a": {"type": "get_var", "var": "p"},
                        "b": {"type": "const", "var": "p", "value": [1.0, 2.0]},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

    def test_compile_trajectory_problem_with_adapter_compiles_runtime(self) -> None:
        adapter = _FakeAdapter(dof=1, order=4)
        dsl = self._linear_dsl(include_q_dim=True)

        compiled = compile_trajectory_problem_with_adapter(
            dsl,
            model=object(),
            data={},
            adapter=adapter,
        )

        assert adapter.validated
        assert adapter.seen_p_var == "p"
        assert float(compiled.prepared.dt) == pytest.approx(0.2, rel=0.0, abs=1e-7)
        assert int(compiled.prepared.model_order) == 4
        assert int(compiled.prepared.trajectory_map.p_dim) == 2
        assert int(compiled.runtime.pack.n_total) == 2

        r, J = compiled.runtime.linearize()
        assert np.allclose(r, np.array([-1.0, -2.0], dtype=float))
        assert np.allclose(J, np.eye(2, dtype=float))

    def test_compile_trajectory_problem_with_adapter_uses_model_dof_as_q_dim_default(self) -> None:
        adapter = _FakeAdapter(dof=1, order=2)
        dsl = self._linear_dsl(include_q_dim=False)

        compiled = compile_trajectory_problem_with_adapter(
            dsl,
            model=object(),
            data={},
            adapter=adapter,
        )

        assert int(compiled.prepared.trajectory_map.q_dim) == 1

