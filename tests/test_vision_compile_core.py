from __future__ import annotations

from rei.core.state_schema import DTYPE_VISION
from rei.optimize.dsl import prepare_vision_calibration_problem_dsl

class TestVisionCompileCore:
    def test_prepare_vision_calibration_problem_dsl_builds_standard_term(self) -> None:
        dsl = {
            "vision": {
                "owner_name": "cam0",
                "observations": [1.0, 2.0, 3.0],
            },
            "variables": [{"name": "theta", "dim": 2, "init": [0.0, 0.0]}],
            "terms": [],
        }
        prepared = prepare_vision_calibration_problem_dsl(dsl)
        assert prepared.p_var == "theta"
        assert prepared.owner_type == "camera"
        assert prepared.owner_name == "cam0"
        assert prepared.field == "reproj"
        assert prepared.k == 0
        assert prepared.term_name == "camera_reproj_error"
        assert prepared.observations.tolist() == [1.0, 2.0, 3.0]

        terms = prepared.dsl["terms"]
        assert len(terms) == 1
        expr = terms[0]["expr"]
        assert expr["name"] == "camera_reproj_error"
        assert expr["a"]["key"]["dtype"] == DTYPE_VISION
        assert expr["a"]["key"]["owner_name"] == "cam0"
        assert expr["a"]["jac"]["var"] == "theta"
        assert expr["b"]["value"] == [1.0, 2.0, 3.0]

    def test_prepare_vision_calibration_problem_dsl_rewrites_existing_named_term(self) -> None:
        dsl = {
            "vision": {
                "owner_name": "cam0",
                "field": "reproj",
                "term_name": "reproj_error",
                "observations": [0.0, 1.0],
            },
            "variables": [{"name": "theta", "dim": 2, "init": [0.1, 0.2]}],
            "terms": [
                {
                    "expr": {"type": "const", "name": "reproj_error", "value": [10.0]},
                    "cost": {"type": "l2"},
                },
                {
                    "expr": {"type": "const", "name": "reproj_error", "value": [20.0]},
                    "cost": {"type": "l2"},
                },
            ],
        }
        prepared = prepare_vision_calibration_problem_dsl(dsl)
        terms = prepared.dsl["terms"]
        assert len(terms) == 1
        expr = terms[0]["expr"]
        assert expr["name"] == "reproj_error"
        assert expr["type"] == "sub"

    def test_prepare_vision_calibration_problem_dsl_can_skip_term_standardization(self) -> None:
        dsl = {
            "vision": {
                "owner_type": "camera",
                "owner_name": "cam0",
                "field": "reproj",
                "observations": [1.0, 2.0],
            },
            "variables": [{"name": "theta", "dim": 2, "init": [0.0, 0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "name": "vision_raw",
                        "key": {
                            "k": 0,
                            "owner_type": "camera",
                            "owner_name": "other_cam",
                            "dtype": DTYPE_VISION,
                            "field": "reproj",
                        },
                        "jac": {"var": "theta"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        prepared = prepare_vision_calibration_problem_dsl(dsl, standardize_terms=False)
        terms = prepared.dsl["terms"]
        assert len(terms) == 1
        key = terms[0]["expr"]["key"]
        assert key["owner_name"] == "cam0"
        assert terms[0]["expr"]["name"] == "vision_raw"

