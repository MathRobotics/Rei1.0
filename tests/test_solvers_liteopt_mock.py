from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from eiopt.core.expr.types import DirectVectorExpr, RuntimeContext, Variable, VariablePack
from eiopt.optimize.costs import L2Cost
from eiopt.problem import NLSProblem
from eiopt.optimize.runtime import NLSRuntime
from eiopt.optimize.solvers import solve


def _build_scalar_runtime(target: float) -> tuple[NLSRuntime, Variable]:
    x_var = Variable(name="x", x=np.array([0.0], dtype=float))
    pack = VariablePack([x_var])

    def value(ctx: RuntimeContext) -> np.ndarray:
        x = float(ctx.pack.vars[0].x[0])
        return np.array([x - float(target)], dtype=float)

    def blocks(_ctx: RuntimeContext) -> list[np.ndarray]:
        return [np.array([[1.0]], dtype=float)]

    expr = DirectVectorExpr(name="x_minus_target", vars=[x_var], fn_value=value, fn_blocks=blocks)
    problem = NLSProblem(variables=pack, terms=[(expr, L2Cost())])
    runtime = NLSRuntime(problem=problem, ctx=RuntimeContext(pack=pack), required=[])
    return runtime, x_var


class TestSolversLiteoptMock:
    def test_solve_dispatches_liteopt_solver_with_stub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, float | int] = {}
        liteopt_mod = types.ModuleType("liteopt")

        def gd(fun, grad, x0, *, step_size=1e-3, max_iters=200, tol_grad=1e-4):
            calls["step_size"] = float(step_size)
            calls["max_iters"] = int(max_iters)
            calls["tol_grad"] = float(tol_grad)
            x = np.asarray(x0, dtype=float).reshape(-1).copy()

            for k in range(int(max_iters)):
                g = np.asarray(grad(x), dtype=float).reshape(-1)
                if float(np.linalg.norm(g)) < float(tol_grad):
                    return x.copy(), float(fun(x)), True, int(k)
                x = x - float(step_size) * g

            return x.copy(), float(fun(x)), False, int(max_iters)

        liteopt_mod.gd = gd
        monkeypatch.setitem(sys.modules, "liteopt", liteopt_mod)

        runtime, x_var = _build_scalar_runtime(target=3.0)
        out = solve(
            runtime,
            solver="liteopt",
            options={"max_iters": 2000, "step_size": 0.1, "tol_grad": 1e-8},
        )
        x_star = out.solution
        stats = out.stats

        assert calls == {"step_size": 0.1, "max_iters": 2000, "tol_grad": 1e-8}
        assert stats.converged
        assert float(stats.initial_objective or 0.0) >= float(stats.objective or 0.0)
        assert float(stats.objective or 0.0) < 1e-12
        assert float(stats.residual_norm or 0.0) < 1e-6
        assert int(stats.iterations) > 0
        assert float(x_star[0]) == pytest.approx(3.0, rel=0.0, abs=1e-6)
        assert float(x_var.x[0]) == pytest.approx(3.0, rel=0.0, abs=1e-6)

    def test_solve_liteopt_raises_import_error_when_module_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime, _x_var = _build_scalar_runtime(target=0.0)
        monkeypatch.setitem(sys.modules, "liteopt", None)
        with pytest.raises(ImportError, match="solve_liteopt_gd requires liteopt"):
            _ = solve(runtime, solver="liteopt")

    def test_solve_liteopt_options_override_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, float | int] = {}
        liteopt_mod = types.ModuleType("liteopt")

        def gd(fun, grad, x0, *, step_size=1e-3, max_iters=200, tol_grad=1e-4):
            del grad
            calls["step_size"] = float(step_size)
            calls["max_iters"] = int(max_iters)
            calls["tol_grad"] = float(tol_grad)
            x = np.asarray(x0, dtype=float).reshape(-1).copy()
            return x.copy(), float(fun(x)), True, 0

        liteopt_mod.gd = gd
        monkeypatch.setitem(sys.modules, "liteopt", liteopt_mod)

        runtime, _x_var = _build_scalar_runtime(target=0.0)
        out = solve(
            runtime,
            solver="liteopt",
            options={
                "max_iters": 2000,
                "step_size": 0.1,
                "tol_grad": 1e-8,
                "backend_options": {"step_size": 0.25, "max_iters": 12, "tol_grad": 5e-3},
            },
        )

        assert out.converged
        assert calls == {"step_size": 0.25, "max_iters": 12, "tol_grad": 5e-3}

    def test_solve_liteopt_accepts_generic_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, float | int] = {}
        liteopt_mod = types.ModuleType("liteopt")

        def gd(fun, grad, x0, *, step_size=1e-3, max_iters=200, tol_grad=1e-4):
            del grad
            calls["step_size"] = float(step_size)
            calls["max_iters"] = int(max_iters)
            calls["tol_grad"] = float(tol_grad)
            x = np.asarray(x0, dtype=float).reshape(-1).copy()
            return x.copy(), float(fun(x)), True, 0

        liteopt_mod.gd = gd
        monkeypatch.setitem(sys.modules, "liteopt", liteopt_mod)

        runtime, _x_var = _build_scalar_runtime(target=0.0)
        out = solve(
            runtime,
            solver="liteopt",
            options={"step_size": 0.2, "max_iters": 15, "tol_grad": 1e-5},
        )

        assert out.converged
        assert calls == {"step_size": 0.2, "max_iters": 15, "tol_grad": 1e-5}
