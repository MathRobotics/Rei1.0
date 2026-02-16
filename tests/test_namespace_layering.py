from __future__ import annotations

import importlib
import unittest

import numpy as np

import eiopt
from eiopt.backends.state.template import BackendDispatchStateBuilder
from eiopt.core.expr.types import DirectVectorExpr, RuntimeContext, Variable, VariablePack
from eiopt.optimize.builder import compile_nls_problem
from eiopt.optimize.costs import L2Cost
from eiopt.optimize.dsl import prepare_trajectory_problem_dsl
from eiopt.optimize.problem import NLSProblem
from eiopt.optimize.reductions import build_nullspace_equality_reduction
from eiopt.optimize.report import format_solve_report
from eiopt.optimize.runtime import NLSRuntime
from eiopt.optimize.simplex_weight_solver import estimate_weights_simplex
from eiopt.optimize.solvers import solve, solve_gauss_newton
from eiopt.optimize.term_gradient_matrix import build_term_gradient_matrix
from eiopt.optimize_backends.trajectory_adapter import compile_trajectory_problem_with_adapter


class TestNamespaceLayering(unittest.TestCase):
    def test_optimize_builder_compile_nls_problem(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [2.0]}],
            "terms": [
                {
                    "expr": {"type": "get_var", "name": "x_identity", "var": "x"},
                    "cost": {"type": "l2"},
                }
            ],
        }

        runtime = compile_nls_problem(
            dsl,
            build_state=lambda *_args, **_kwargs: {},
        )
        r, J = runtime.linearize()
        self.assertTrue(np.allclose(r, np.array([2.0], dtype=float)))
        self.assertTrue(np.allclose(J, np.array([[1.0]], dtype=float)))

    def test_canonical_types_construct_runtime(self) -> None:
        x_var = Variable(name="x", x=np.array([0.0], dtype=float))
        pack = VariablePack([x_var])

        def value(ctx: RuntimeContext) -> np.ndarray:
            x = float(ctx.pack.vars[0].x[0])
            return np.array([x - 1.0], dtype=float)

        def blocks(_ctx: RuntimeContext) -> list[np.ndarray]:
            return [np.array([[1.0]], dtype=float)]

        expr = DirectVectorExpr(name="x_minus_1", vars=[x_var], fn_value=value, fn_blocks=blocks)
        problem = NLSProblem(variables=pack, terms=[(expr, L2Cost())])
        runtime = NLSRuntime(problem=problem, ctx=RuntimeContext(pack=pack), required=[])

        x_star, cost0, cost, _iters, _rnorm, _dxnorm, converged = solve_gauss_newton(runtime, max_iters=8)
        self.assertTrue(converged)
        self.assertGreaterEqual(cost0, cost)
        self.assertLess(cost, 1e-20)
        self.assertAlmostEqual(float(x_star[0]), 1.0, places=10)

    def test_canonical_solver_entrypoint(self) -> None:
        self.assertTrue(callable(solve))

    def test_canonical_entrypoints_exist(self) -> None:
        self.assertTrue(callable(prepare_trajectory_problem_dsl))
        self.assertTrue(callable(build_nullspace_equality_reduction))
        self.assertTrue(callable(format_solve_report))
        self.assertTrue(callable(build_term_gradient_matrix))
        self.assertTrue(callable(estimate_weights_simplex))
        self.assertTrue(callable(compile_trajectory_problem_with_adapter))
        self.assertTrue(issubclass(BackendDispatchStateBuilder, object))

    def test_removed_top_level_legacy_aliases(self) -> None:
        self.assertFalse(hasattr(eiopt, "compile_problem"))
        self.assertFalse(hasattr(eiopt, "ProblemRuntime"))
        self.assertFalse(hasattr(eiopt, "dsl"))
        self.assertFalse(hasattr(eiopt, "model"))
        self.assertFalse(hasattr(eiopt, "solvers"))
        self.assertFalse(hasattr(eiopt, "expr"))

    def test_removed_legacy_namespace_modules(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("eiopt.dsl")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("eiopt.model")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("eiopt.solvers")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("eiopt.expr")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends.trajectory_adapter")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends.kots")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends.pinocchio")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends._template")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends._spatial")

    def test_removed_optimize_legacy_aliases(self) -> None:
        optimize_builder = importlib.import_module("eiopt.optimize.builder")
        optimize_problem = importlib.import_module("eiopt.optimize.problem")
        self.assertFalse(hasattr(optimize_builder, "build_problem"))
        self.assertFalse(hasattr(optimize_builder, "collect_required"))
        self.assertFalse(hasattr(optimize_problem, "Problem"))


if __name__ == "__main__":
    unittest.main()
