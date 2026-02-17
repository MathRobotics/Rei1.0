from __future__ import annotations

import copy
import unittest

import numpy as np

from eiopt.optimize.builder import compile_nls_problem
from eiopt.optimize.dsl import split_terms_by_component


class TestDslComponentSplit(unittest.TestCase):
    def test_component_expr_selects_rows_and_jacobian(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 6, "init": [1, 2, 3, 4, 5, 6]}],
            "terms": [
                {
                    "expr": {
                        "type": "component",
                        "name": "x_j1",
                        "segment_dim": 2,
                        "index": 1,
                        "base": {"type": "get_var", "name": "x_all", "var": "x"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        terms = runtime.linearize_terms(weighted=False)
        self.assertEqual(len(terms), 1)

        r = np.asarray(terms[0].residual, dtype=float).reshape(-1)
        J = np.asarray(terms[0].jacobian, dtype=float)
        self.assertTrue(np.allclose(r, np.array([2.0, 4.0, 6.0], dtype=float)))

        expected = np.zeros((3, 6), dtype=float)
        expected[0, 1] = 1.0
        expected[1, 3] = 1.0
        expected[2, 5] = 1.0
        self.assertTrue(np.allclose(J, expected))

    def test_split_terms_by_component_preserves_cost_for_scalar_and_diag(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 6, "init": [1, 2, 3, 4, 5, 6]}],
            "terms": [
                {
                    "expr": {"type": "get_var", "name": "x_scalar", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 2.0},
                },
                {
                    "expr": {"type": "get_var", "name": "x_diag", "var": "x"},
                    "cost": {"type": "diag_weight", "w": [1, 2, 3, 4, 5, 6]},
                },
            ],
        }
        runtime_orig = compile_nls_problem(copy.deepcopy(dsl), build_state=lambda *_args, **_kwargs: {})
        cost_orig = float(runtime_orig.cost_value())

        dsl_split = copy.deepcopy(dsl)
        n_expanded = split_terms_by_component(dsl_split, segment_dim=2, term_indices=[0, 1])
        self.assertEqual(n_expanded, 2)
        self.assertEqual(len(dsl_split["terms"]), 4)

        names = [str(t["expr"]["name"]) for t in dsl_split["terms"]]
        self.assertEqual(names, ["x_scalar_j0", "x_scalar_j1", "x_diag_j0", "x_diag_j1"])
        self.assertEqual(dsl_split["terms"][2]["cost"]["w"], [1.0, 3.0, 5.0])
        self.assertEqual(dsl_split["terms"][3]["cost"]["w"], [2.0, 4.0, 6.0])
        self.assertEqual(dsl_split["terms"][0]["attrs"]["joint_component"], 0)
        self.assertEqual(dsl_split["terms"][1]["attrs"]["joint_component"], 1)

        runtime_split = compile_nls_problem(dsl_split, build_state=lambda *_args, **_kwargs: {})
        cost_split = float(runtime_split.cost_value())
        self.assertAlmostEqual(cost_orig, cost_split, places=12)


if __name__ == "__main__":
    unittest.main()
