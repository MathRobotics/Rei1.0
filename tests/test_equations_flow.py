from __future__ import annotations

import numpy as np
import pytest

from eiopt.equations import as_linear_equation_problem
from eiopt.flow import as_constraint_problem, as_project_problem
from eiopt.optimize.builder import compile_nls_problem


class TestEquationFlowAdapters:
    def test_as_linear_equation_problem_matches_runtime_linearize(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [2.0]}],
            "terms": [
                {
                    "expr": {"type": "get_var", "name": "x_identity", "var": "x"},
                    "cost": {"type": "l2"},
                }
            ],
        }
        runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        eq_problem = as_linear_equation_problem(runtime)

        r_rt, J_rt = runtime.linearize()
        r_eq, J_eq = eq_problem.linearize()

        assert np.allclose(r_eq, r_rt)
        assert np.allclose(J_eq, J_rt)
        assert np.allclose(eq_problem.eval(), r_rt)

    def test_as_constraint_problem_extracts_eq_terms(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [0.25]}],
            "terms": [
                {
                    "constraint": {"kind": "eq"},
                    "expr": {
                        "type": "sub",
                        "name": "eq_to_half",
                        "a": {"type": "get_var", "var": "x"},
                        "b": {"type": "const", "var": "x", "value": [0.5]},
                    },
                    "cost": {"type": "l2"},
                },
                {
                    "expr": {"type": "get_var", "name": "x_reg", "var": "x"},
                    "cost": {"type": "l2"},
                },
            ],
        }
        runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        cprob = as_constraint_problem(runtime, kind="eq", weighted=False)

        g = cprob.constraint()
        Jg = cprob.jacobian_constraint()

        assert g.shape == (1,)
        assert Jg.shape == (1, 1)
        assert np.allclose(g, np.array([-0.25], dtype=float))
        assert np.allclose(Jg, np.array([[1.0]], dtype=float))

    def test_as_project_problem_adds_identity_projection(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [0.0]}],
            "terms": [
                {
                    "expr": {"type": "get_var", "name": "x_identity", "var": "x"},
                    "cost": {"type": "l2"},
                }
            ],
        }
        runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        pprob = as_project_problem(runtime)

        x = np.array([1.25], dtype=float)
        assert np.allclose(pprob.project(x), x)

        pprob.set_point(x)
        x_after = np.asarray(runtime.pack.get(), dtype=float).reshape(-1)
        assert np.allclose(x_after, x)

        with pytest.raises(ValueError, match="expected size"):
            pprob.project(np.array([1.0, 2.0], dtype=float))
