from __future__ import annotations

import numpy as np
import pytest

from rei.optimize.builder import compile_nls_problem
from rei.optimize.solver_problem import as_solver_problem
from rei.optimize.solvers import solve, solve_gauss_newton
from rei.problem import NLSRuntimeConstraintProblem, as_linearized_problem


class TestProblemCapabilities:
    def test_as_solver_problem_exposes_external_solver_callables(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "name": "x_to_2",
                        "a": {"type": "get_var", "var": "x"},
                        "b": {"type": "const", "var": "x", "value": [2.0]},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        problem = as_solver_problem(runtime)

        assert problem.n_total == 1
        assert np.allclose(problem.x0, np.array([0.0], dtype=float))
        assert np.allclose(problem.residual(np.array([0.5], dtype=float)), np.array([-1.5], dtype=float))
        assert np.allclose(problem.jacobian(np.array([0.5], dtype=float)), np.array([[1.0]], dtype=float))
        assert problem.objective(np.array([0.5], dtype=float)) == pytest.approx(2.25)
        assert np.allclose(problem.gradient(np.array([0.5], dtype=float)), np.array([-3.0], dtype=float))
        f, g = problem.objective_and_gradient(np.array([2.0], dtype=float))
        assert f == pytest.approx(0.0)
        assert np.allclose(g, np.array([0.0], dtype=float))

    def test_as_linearized_problem_matches_runtime_linearize(self) -> None:
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
        adapter = as_linearized_problem(runtime)

        r_rt, J_rt = runtime.linearize()
        r_ad, J_ad = adapter.linearize()

        assert np.allclose(r_ad, r_rt)
        assert np.allclose(J_ad, J_rt)
        assert np.allclose(adapter.eval(), r_rt)

    def test_as_linearized_problem_respects_weight_and_term_selection(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [2.0]}],
            "terms": [
                {
                    "expr": {"type": "get_var", "name": "t_raw", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 4.0},
                },
                {
                    "expr": {
                        "type": "sub",
                        "name": "t_sel",
                        "a": {"type": "get_var", "var": "x"},
                        "b": {"type": "const", "var": "x", "value": [1.0]},
                    },
                    "cost": {"type": "scalar_weight", "w": 3.0},
                },
            ],
        }
        runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})

        adapter = as_linearized_problem(runtime, weighted=False, term_indices=(1,))
        r_ref, J_ref = runtime.linearize_stacked_terms(weighted=False, term_indices=[1])
        r_ad, J_ad = adapter.linearize()

        assert np.allclose(r_ad, r_ref)
        assert np.allclose(J_ad, J_ref)

    def test_solve_gauss_newton_accepts_linearized_problem_adapter(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "name": "x_to_2_5",
                        "a": {"type": "get_var", "var": "x"},
                        "b": {"type": "const", "var": "x", "value": [2.5]},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        adapter = as_linearized_problem(runtime)

        out = solve_gauss_newton(
            adapter,
            max_iters=16,
            tol_r=1e-14,
            tol_dx=1e-14,
        )
        x_star = out.solution
        stats = out.stats

        assert stats.converged
        assert float(stats.initial_objective or 0.0) >= float(stats.objective or 0.0)
        assert float(stats.objective or 0.0) < 1e-20
        assert float(x_star[0]) == pytest.approx(2.5, rel=0.0, abs=1e-10)

    def test_solve_gauss_newton_x0_overrides_adapter_current_point(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "name": "x_to_2_5",
                        "a": {"type": "get_var", "var": "x"},
                        "b": {"type": "const", "var": "x", "value": [2.5]},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        adapter = as_linearized_problem(runtime)

        out = solve_gauss_newton(
            adapter,
            x0=np.array([1.0], dtype=float),
            max_iters=16,
            tol_r=1e-14,
            tol_dx=1e-14,
        )

        assert out.converged
        assert float(out.stats.initial_objective or 0.0) == pytest.approx(2.25, rel=0.0, abs=1e-14)
        assert np.allclose(np.asarray(out.meta["x0"], dtype=float), np.array([1.0], dtype=float))
        assert float(out.solution[0]) == pytest.approx(2.5, rel=0.0, abs=1e-10)

    def test_dispatch_solve_accepts_linearized_problem_adapter(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "name": "x_to_1",
                        "a": {"type": "get_var", "var": "x"},
                        "b": {"type": "const", "var": "x", "value": [1.0]},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        adapter = as_linearized_problem(runtime)

        out = solve(
            adapter,
            solver="gauss_newton",
            options={"max_iters": 12, "tol_r": 1e-14, "tol_dx": 1e-14},
        )
        x_star = out.solution

        assert out.converged
        assert float(x_star[0]) == pytest.approx(1.0, rel=0.0, abs=1e-10)

    def test_constraint_problem_adapter_extracts_eq_terms(self) -> None:
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
        cprob = NLSRuntimeConstraintProblem(runtime, kind="eq", weighted=False)

        g = cprob.constraint()
        Jg = cprob.jacobian_constraint()

        assert g.shape == (1,)
        assert Jg.shape == (1, 1)
        assert np.allclose(g, np.array([-0.25], dtype=float))
        assert np.allclose(Jg, np.array([[1.0]], dtype=float))
