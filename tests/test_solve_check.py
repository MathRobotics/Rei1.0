from __future__ import annotations

import numpy as np

from eiopt.optimize.builder import compile_nls_problem
from eiopt.optimize.kkt import check_kkt_conditions, check_kkt_residuals

class TestSolveCheck:
    def test_check_kkt_conditions_unconstrained_stationarity(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "a": {"type": "get_var", "var": "x"},
                        "b": {"type": "const", "var": "x", "value": [3.0]},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        x_cur = runtime.pack.get().copy()
        runtime.pack.apply_dx(np.array([3.0], dtype=float) - x_cur)

        out = check_kkt_conditions(runtime)
        assert out.ok
        assert out.stationarity_inf < 1e-10
        assert out.n_eq_rows == 0
        assert out.n_ineq_rows == 0

    def test_check_kkt_conditions_eq_constraint(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 2, "init": [1.0, 1.0]}],
            "terms": [
                {
                    "expr": {"type": "get_var", "var": "x"},
                    "cost": {"type": "l2"},
                },
                {
                    "constraint": {"kind": "eq"},
                    "expr": {
                        "type": "sub",
                        "a": {"type": "get_var", "var": "x"},
                        "b": {"type": "const", "var": "x", "value": {"fill": 1.0}},
                    },
                    "cost": {"type": "l2"},
                },
            ],
        }
        runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})

        out = check_kkt_conditions(runtime, stationarity_tol=1e-10, eq_tol=1e-10)
        assert out.ok
        assert out.stationarity_inf < 1e-10
        assert out.eq_violation_inf < 1e-10
        assert out.n_eq_rows == 2
        assert np.allclose(out.lambda_eq, np.array([-1.0, -1.0], dtype=float), atol=1e-10)

    def test_check_kkt_conditions_ineq_violation_detected(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [2.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "a": {"type": "get_var", "var": "x"},
                        "b": {"type": "const", "var": "x", "value": [2.0]},
                    },
                    "cost": {"type": "l2"},
                },
                {
                    "constraint": {"kind": "ineq"},
                    "expr": {
                        "type": "sub",
                        "a": {"type": "get_var", "var": "x"},
                        "b": {"type": "const", "var": "x", "value": [1.0]},
                    },
                    "cost": {"type": "l2"},
                },
            ],
        }
        runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})

        out = check_kkt_conditions(runtime, ineq_tol=1e-12)
        assert not (out.ok)
        assert out.ineq_violation_inf > 0.5
        assert "ineq_violation_inf" in out.message

    def test_check_kkt_residuals_generic(self) -> None:
        out = check_kkt_residuals(
            grad_objective=np.array([1.0, 1.0], dtype=float),
            eq_residual=np.zeros((1,), dtype=float),
            eq_jacobian=np.array([[1.0, 1.0]], dtype=float),
            ineq_residual=np.zeros((0,), dtype=float),
            ineq_jacobian=np.zeros((0, 2), dtype=float),
        )
        assert out.ok
        assert out.stationarity_inf < 1e-12
        assert out.n_eq_rows == 1
        assert out.n_ineq_rows == 0

