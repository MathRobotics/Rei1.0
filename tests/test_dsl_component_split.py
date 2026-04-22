from __future__ import annotations

import pytest

import copy

import numpy as np

from rei.optimize.builder import compile_nls_problem
from rei.optimize.dsl import split_terms_by_component

class TestDslComponentSplit:
    def test_component_expr_infers_segment_dim_from_time_chunked_get_var(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.1},
            "variables": [{"name": "x", "dim": 6, "init": [1, 2, 3, 4, 5, 6]}],
            "terms": [
                {
                    "expr": {
                        "type": "component",
                        "name": "x_j1",
                        "index": 1,
                        "base": {"type": "get_var", "name": "x_all", "var": "x"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        terms = runtime.linearize_terms(weighted=False)
        assert len(terms) == 1

        r = np.asarray(terms[0].residual, dtype=float).reshape(-1)
        J = np.asarray(terms[0].jacobian, dtype=float)
        assert np.allclose(r, np.array([2.0, 4.0, 6.0], dtype=float))

        expected = np.zeros((3, 6), dtype=float)
        expected[0, 1] = 1.0
        expected[1, 3] = 1.0
        expected[2, 5] = 1.0
        assert np.allclose(J, expected)

    def test_component_expr_supports_auto_segment_dim_on_total_joint_state(self) -> None:
        dsl = {
            "time": {"N": 1, "dt": 0.1},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 2,
                "A": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            },
            "variables": [{"name": "p", "dim": 2, "init": [0.0, 0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "component",
                        "name": "q_init_j1",
                        "segment_dim": "auto",
                        "index": 1,
                        "base": {
                            "type": "sub",
                            "a": {
                                "type": "get_state",
                                "key": {
                                    "k": 0,
                                    "owner_type": "total_joint",
                                    "owner_name": "robot",
                                    "dtype": "coord",
                                    "field": "q",
                                },
                                "jac": {"var": "p"},
                            },
                            "b": {
                                "type": "const",
                                "var": "p",
                                "value": {"fill": 0.0},
                            },
                        },
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        def build_state(_x_all: np.ndarray, *, required=None, **_kwargs):
            req = set(required or [])
            out = {}
            for key in req:
                if key.owner.owner_type != "total_joint" or key.owner.owner_name != "robot":
                    continue
                if key.dtype == "coord" and key.field == "q":
                    out[key] = np.array([1.5, -0.25], dtype=float)
                elif key.dtype == "coord" and key.field == "q_J_p":
                    out[key] = np.eye(2, dtype=float)
            return out

        runtime = compile_nls_problem(dsl, build_state=build_state)
        terms = runtime.linearize_terms(weighted=False)
        assert len(terms) == 1

        r = np.asarray(terms[0].residual, dtype=float).reshape(-1)
        J = np.asarray(terms[0].jacobian, dtype=float)
        assert np.allclose(r, np.array([-0.25], dtype=float))
        assert np.allclose(J, np.array([[0.0, 1.0]], dtype=float))

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
        assert n_expanded == 2
        assert len(dsl_split["terms"]) == 4

        names = [str(t["expr"]["name"]) for t in dsl_split["terms"]]
        assert names == ["x_scalar_j0", "x_scalar_j1", "x_diag_j0", "x_diag_j1"]
        assert dsl_split["terms"][2]["cost"]["w"] == [1.0, 3.0, 5.0]
        assert dsl_split["terms"][3]["cost"]["w"] == [2.0, 4.0, 6.0]
        assert dsl_split["terms"][0]["attrs"]["joint_component"] == 0
        assert dsl_split["terms"][1]["attrs"]["joint_component"] == 1

        runtime_split = compile_nls_problem(dsl_split, build_state=lambda *_args, **_kwargs: {})
        cost_split = float(runtime_split.cost_value())
        assert cost_orig == pytest.approx(cost_split, rel=0.0, abs=10 ** (-(12)))
