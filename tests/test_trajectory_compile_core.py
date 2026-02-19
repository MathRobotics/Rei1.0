from __future__ import annotations

import pytest

from eiopt.optimize.dsl.trajectory_compile import prepare_trajectory_problem_dsl

class TestTrajectoryCompileCore:
    def _linear_dsl(self) -> dict:
        return {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 2,
                "A": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "terms": [],
        }

    def test_prepare_trajectory_problem_dsl_keeps_input_immutable(self) -> None:
        dsl = self._linear_dsl()
        assert "variables" not in dsl

        prepared = prepare_trajectory_problem_dsl(
            dsl,
            model_dof=2,
            model_order=3,
        )

        assert "variables" not in dsl
        assert prepared.p_var == "p"
        assert prepared.dt == pytest.approx(0.2, rel=0.0, abs=1e-7)
        assert prepared.model_order == 3
        assert sorted(prepared.trajectory_derivative_maps.keys()) == [0, 1, 2]

        variables = prepared.dsl.get("variables", [])
        assert len(variables) == 1
        assert variables[0]["name"] == "p"
        assert variables[0]["dim"] == 4
        assert variables[0]["init"] == [0.0, 0.0, 0.0, 0.0]

    def test_prepare_trajectory_problem_dsl_rejects_negative_max_derivative_order(self) -> None:
        dsl = self._linear_dsl()
        with pytest.raises(ValueError, match="max_derivative_order must be >= 0"):
            _ = prepare_trajectory_problem_dsl(
                dsl,
                model_order=3,
                max_derivative_order=-1,
            )

    def test_prepare_trajectory_problem_dsl_allows_explicit_p_var_override(self) -> None:
        dsl = self._linear_dsl()

        prepared = prepare_trajectory_problem_dsl(
            dsl,
            p_var="z",
            model_order=2,
        )

        assert prepared.p_var == "z"
        variables = prepared.dsl.get("variables", [])
        assert len(variables) == 1
        assert variables[0]["name"] == "z"
        assert variables[0]["dim"] == 4

