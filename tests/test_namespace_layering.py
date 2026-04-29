from __future__ import annotations

import pytest

import importlib

import numpy as np

import rei
from rei.backends.state.dispatch.template import BackendDispatchStateBuilder
from rei.core.expr.types import DirectVectorExpr, RuntimeContext, Variable, VariablePack
from rei.equations import as_linear_equation_problem
from rei.flow import as_constraint_problem, as_project_problem
from rei.equations import (
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
from rei.optimize.builder import compile_nls_problem, compile_nls_problem_spec
from rei.optimize.costs import L2Cost
from rei.optimize.dsl import prepare_trajectory_problem_dsl
from rei.problem import NLSProblem
from rei.optimize.reductions import build_nullspace_equality_reduction
from rei.optimize.report import format_solve_report
from rei.optimize.runtime import NLSRuntime
from rei.optimize.solvers import solve, solve_gauss_newton
from rei.optimize.term_gradient_matrix import build_term_gradient_matrix
from rei.optimize_backends.trajectory_adapter import compile_trajectory_problem_with_adapter

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

        out = solve_gauss_newton(runtime, max_iters=8)
        x_star = out.solution
        stats = out.stats
        assert stats.converged
        assert float(stats.initial_objective or 0.0) >= float(stats.objective or 0.0)
        assert float(stats.objective or 0.0) < 1e-20
        assert float(x_star[0]) == pytest.approx(1.0, rel=0.0, abs=10 ** (-(10)))

    def test_canonical_solver_entrypoint(self) -> None:
        assert callable(solve)

    def test_canonical_entrypoints_exist(self) -> None:
        assert callable(prepare_trajectory_problem_dsl)
        assert callable(compile_nls_problem_spec)
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
        assert not (hasattr(rei, "compile_problem"))
        assert not (hasattr(rei, "ProblemRuntime"))
        assert not (hasattr(rei, "dsl"))
        assert not (hasattr(rei, "model"))
        assert not (hasattr(rei, "solvers"))
        assert not (hasattr(rei, "expr"))

    def test_removed_legacy_namespace_modules(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.dsl")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.model")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.solvers")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.expr")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.inference")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.feasibility")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.backends.trajectory_adapter")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.backends.kots")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.backends.pinocchio")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.backends._template")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.backends._spatial")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.backends.state.template")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.backends.state.composite")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.backends.state.spatial")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.backends.state.kots")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.backends.state.pinocchio")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.backends.state.vision_pinhole")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.optimize.problem")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.optimize.simplex_weight_solver")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.optimize._xops")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rei.simplex_weight_solver")

    def test_removed_optimize_legacy_aliases(self) -> None:
        optimize_builder = importlib.import_module("rei.optimize.builder")
        assert not (hasattr(optimize_builder, "build_problem"))
        assert not (hasattr(optimize_builder, "collect_required"))
