from __future__ import annotations

import unittest

from eiopt.core.state_schema import DTYPE_VISION
from eiopt.optimize.dsl import prepare_vision_calibration_problem_dsl


class TestVisionCompileCore(unittest.TestCase):
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
        self.assertEqual(prepared.p_var, "theta")
        self.assertEqual(prepared.owner_type, "camera")
        self.assertEqual(prepared.owner_name, "cam0")
        self.assertEqual(prepared.field, "reproj")
        self.assertEqual(prepared.k, 0)
        self.assertEqual(prepared.term_name, "camera_reproj_error")
        self.assertEqual(prepared.observations.tolist(), [1.0, 2.0, 3.0])

        terms = prepared.dsl["terms"]
        self.assertEqual(len(terms), 1)
        expr = terms[0]["expr"]
        self.assertEqual(expr["name"], "camera_reproj_error")
        self.assertEqual(expr["a"]["key"]["dtype"], DTYPE_VISION)
        self.assertEqual(expr["a"]["key"]["owner_name"], "cam0")
        self.assertEqual(expr["a"]["jac"]["var"], "theta")
        self.assertEqual(expr["b"]["value"], [1.0, 2.0, 3.0])

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
        self.assertEqual(len(terms), 1)
        expr = terms[0]["expr"]
        self.assertEqual(expr["name"], "reproj_error")
        self.assertEqual(expr["type"], "sub")

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
        self.assertEqual(len(terms), 1)
        key = terms[0]["expr"]["key"]
        self.assertEqual(key["owner_name"], "cam0")
        self.assertEqual(terms[0]["expr"]["name"], "vision_raw")


if __name__ == "__main__":
    unittest.main()
