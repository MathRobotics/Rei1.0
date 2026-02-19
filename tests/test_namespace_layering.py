from __future__ import annotations

import pytest

import importlib

import numpy as np

import eiopt
from eiopt.backends.state.dispatch.template import BackendDispatchStateBuilder
from eiopt.core.expr.types import DirectVectorExpr, RuntimeContext, Variable, VariablePack
from eiopt.equations import as_linear_equation_problem
from eiopt.flow import as_constraint_problem, as_project_problem
from eiopt.equations import (
    RuntimeStationaritySource,
    SimplexMinNormProblem,
    build_reference_simplex_init,
    build_stationarity_gradient_matrix,
    filter_stationarity_contributions,
    normalize_simplex_nonnegative,
    select_active_stationarity_indices,
    solve_projected_linearized_min_norm,
    solve_simplex_min_norm,
    term_constraint_kind,
)
from eiopt.optimize.builder import compile_nls_problem
from eiopt.optimize.costs import L2Cost
from eiopt.optimize.dsl import prepare_trajectory_problem_dsl
from eiopt.problem import NLSProblem
from eiopt.optimize.reductions import build_nullspace_equality_reduction
from eiopt.optimize.report import format_solve_report
from eiopt.optimize.runtime import NLSRuntime
from eiopt.optimize.solvers import solve, solve_gauss_newton
from eiopt.optimize.term_gradient_matrix import build_term_gradient_matrix
from eiopt.optimize_backends.trajectory_adapter import compile_trajectory_problem_with_adapter

class TestNamespaceLayering:
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
        assert np.allclose(r, np.array([2.0], dtype=float))
        assert np.allclose(J, np.array([[1.0]], dtype=float))

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
        assert converged
        assert cost0 >= cost
        assert cost < 1e-20
        assert float(x_star[0]) == pytest.approx(1.0, rel=0.0, abs=10 ** (-(10)))

    def test_canonical_solver_entrypoint(self) -> None:
        assert callable(solve)

    def test_canonical_entrypoints_exist(self) -> None:
        assert callable(prepare_trajectory_problem_dsl)
        assert callable(build_nullspace_equality_reduction)
        assert callable(format_solve_report)
        assert callable(build_term_gradient_matrix)
        assert SimplexMinNormProblem is not None
        assert callable(solve_projected_linearized_min_norm)
        assert callable(solve_simplex_min_norm)
        assert RuntimeStationaritySource is not None
        assert callable(normalize_simplex_nonnegative)
        assert callable(term_constraint_kind)
        assert callable(filter_stationarity_contributions)
        assert callable(build_stationarity_gradient_matrix)
        assert callable(select_active_stationarity_indices)
        assert callable(build_reference_simplex_init)
        assert callable(as_linear_equation_problem)
        assert callable(as_constraint_problem)
        assert callable(as_project_problem)
        assert callable(compile_trajectory_problem_with_adapter)
        assert issubclass(BackendDispatchStateBuilder, object)

    def test_removed_top_level_legacy_aliases(self) -> None:
        assert not (hasattr(eiopt, "compile_problem"))
        assert not (hasattr(eiopt, "ProblemRuntime"))
        assert not (hasattr(eiopt, "dsl"))
        assert not (hasattr(eiopt, "model"))
        assert not (hasattr(eiopt, "solvers"))
        assert not (hasattr(eiopt, "expr"))

    def test_removed_legacy_namespace_modules(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.dsl")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.model")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.solvers")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.expr")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.inference")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.feasibility")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends.trajectory_adapter")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends.kots")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends.pinocchio")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends._template")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends._spatial")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends.state.template")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends.state.composite")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends.state.spatial")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends.state.kots")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends.state.pinocchio")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.backends.state.vision_pinhole")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.optimize.problem")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.optimize.simplex_weight_solver")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.optimize._xops")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("eiopt.simplex_weight_solver")

    def test_removed_optimize_legacy_aliases(self) -> None:
        optimize_builder = importlib.import_module("eiopt.optimize.builder")
        assert not (hasattr(optimize_builder, "build_problem"))
        assert not (hasattr(optimize_builder, "collect_required"))
