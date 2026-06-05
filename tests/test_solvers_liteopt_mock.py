from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from rei.core.expr.types import DirectVectorExpr, RuntimeContext, Variable, VariablePack
from rei.optimize.costs import L2Cost
from rei.problem import NLSProblem
from rei.optimize.runtime import NLSRuntime
from rei.optimize.solvers import solve


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

    def test_solve_liteopt_gd_supports_options_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, object] = {}
        liteopt_mod = types.ModuleType("liteopt")

        def gd(fun, grad, x0, *, options=None, debug=None):
            del grad, debug
            calls["options"] = dict(options or {})
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
        assert calls["options"] == {"step_size": 0.2, "max_iters": 15, "tol_grad": 1e-5}

    def test_solve_liteopt_gd_accepts_line_search_option(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, object] = {}
        liteopt_mod = types.ModuleType("liteopt")

        def gd(
            fun,
            grad,
            x0,
            *,
            step_size=1e-3,
            max_iters=200,
            tol_grad=1e-4,
            line_search=None,
            verbose=None,
        ):
            del fun, grad
            calls["step_size"] = float(step_size)
            calls["max_iters"] = int(max_iters)
            calls["tol_grad"] = float(tol_grad)
            calls["line_search"] = line_search
            calls["verbose"] = verbose
            x = np.asarray(x0, dtype=float).reshape(-1).copy()
            return x.copy(), 0.0, True, 0

        liteopt_mod.gd = gd
        monkeypatch.setitem(sys.modules, "liteopt", liteopt_mod)

        runtime, _x_var = _build_scalar_runtime(target=0.0)
        out = solve(
            runtime,
            solver="liteopt",
            options={
                "step_size": 0.2,
                "max_iters": 15,
                "tol_grad": 1e-5,
                "line_search": True,
                "verbose": True,
            },
        )

        assert out.converged
        assert calls == {
            "step_size": 0.2,
            "max_iters": 15,
            "tol_grad": 1e-5,
            "line_search": True,
            "verbose": True,
        }

    def test_solve_liteopt_gn_supports_line_search_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, object] = {}
        liteopt_mod = types.ModuleType("liteopt")

        def gn(
            residual,
            jacobian,
            x0,
            *,
            max_iters=None,
            tol_r=None,
            tol_dx=None,
            lambda_=None,
            damping_update=None,
            linear_system=None,
            line_search_method=None,
            line_search=None,
            ls_beta=None,
            ls_min_step=None,
            ls_max_steps=None,
            verbose=None,
        ):
            calls["max_iters"] = int(max_iters)
            calls["tol_r"] = float(tol_r)
            calls["tol_dx"] = float(tol_dx)
            calls["lambda_"] = float(lambda_)
            calls["damping_update"] = str(damping_update)
            calls["linear_system"] = str(linear_system)
            calls["line_search_method"] = str(line_search_method)
            calls["line_search"] = bool(line_search)
            calls["ls_beta"] = float(ls_beta)
            calls["ls_min_step"] = float(ls_min_step)
            calls["ls_max_steps"] = int(ls_max_steps)
            calls["verbose"] = bool(verbose)
            x = np.asarray(x0, dtype=float).reshape(-1).copy()
            _ = np.asarray(residual(x), dtype=float).reshape(-1)
            _ = np.asarray(jacobian(x), dtype=float)
            return [3.0], 0.0, 5, 0.0, 0.0, True

        liteopt_mod.gn = gn
        monkeypatch.setitem(sys.modules, "liteopt", liteopt_mod)

        runtime, x_var = _build_scalar_runtime(target=3.0)
        out = solve(
            runtime,
            solver="liteopt",
            options={
                "method": "gn",
                "max_iters": 50,
                "tol_r": 1e-9,
                "tol_dx": 1e-10,
                "lambda_": 1e-8,
                "damping_update": "fixed",
                "linear_system": "normal_jtj",
                "line_search_method": "strict_decrease",
                "line_search": True,
                "ls_beta": 0.5,
                "ls_min_step": 1e-8,
                "ls_max_steps": 12,
                "verbose": True,
            },
        )

        assert out.converged
        assert out.meta["solver"] == "liteopt_gn"
        assert float(out.solution[0]) == pytest.approx(3.0, rel=0.0, abs=1e-12)
        assert float(x_var.x[0]) == pytest.approx(3.0, rel=0.0, abs=1e-12)
        assert calls["max_iters"] == 50
        assert calls["tol_r"] == pytest.approx(1e-9, rel=0.0, abs=0.0)
        assert calls["tol_dx"] == pytest.approx(1e-10, rel=0.0, abs=0.0)
        assert calls["lambda_"] == pytest.approx(1e-8, rel=0.0, abs=0.0)
        assert calls["damping_update"] == "fixed"
        assert calls["linear_system"] == "normal_jtj"
        assert calls["line_search_method"] == "strict_decrease"
        assert calls["line_search"] is True
        assert calls["ls_beta"] == pytest.approx(0.5, rel=0.0, abs=0.0)
        assert calls["ls_min_step"] == pytest.approx(1e-8, rel=0.0, abs=0.0)
        assert calls["ls_max_steps"] == 12
        assert calls["verbose"] is True

    def test_solve_liteopt_gn_supports_options_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, object] = {}
        liteopt_mod = types.ModuleType("liteopt")

        def gn(residual, x0=None, *, jacobian=None, options=None, debug=None):
            del debug
            calls["options"] = dict(options or {})
            x = np.asarray(x0, dtype=float).reshape(-1).copy()
            _ = np.asarray(residual(x), dtype=float).reshape(-1)
            assert jacobian is not None
            _ = np.asarray(jacobian(x), dtype=float)
            return [3.0], 0.0, 5, 0.0, 0.0, True

        liteopt_mod.gn = gn
        monkeypatch.setitem(sys.modules, "liteopt", liteopt_mod)

        runtime, _x_var = _build_scalar_runtime(target=3.0)
        out = solve(
            runtime,
            solver="liteopt",
            options={
                "method": "gn",
                "max_iters": 50,
                "tol_r": 1e-9,
                "tol_dx": 1e-10,
                "lambda_": 1e-8,
            },
        )

        assert out.converged
        options = calls["options"]
        assert isinstance(options, dict)
        assert options["max_iters"] == 50
        assert options["tol_r"] == pytest.approx(1e-9, rel=0.0, abs=0.0)
        assert options["tol_dx"] == pytest.approx(1e-10, rel=0.0, abs=0.0)
        assert options["lambda_"] == pytest.approx(1e-8, rel=0.0, abs=0.0)
        assert options["damping_update"] == "fixed"
        assert options["linear_system"] == "normal_jtj"
        assert options["line_search_method"] == "strict_decrease"
        assert options["line_search"] is True

    def test_solve_liteopt_gn_emits_on_iter_progress(self, monkeypatch: pytest.MonkeyPatch) -> None:
        liteopt_mod = types.ModuleType("liteopt")

        def gn(residual, jacobian, x0, **kwargs):
            del kwargs
            x0v = np.asarray(x0, dtype=float).reshape(-1)
            x1v = x0v + np.array([0.5], dtype=float)
            _ = np.asarray(residual(x0v), dtype=float).reshape(-1)
            _ = np.asarray(jacobian(x0v), dtype=float)
            _ = np.asarray(residual(x1v), dtype=float).reshape(-1)
            _ = np.asarray(jacobian(x1v), dtype=float)
            return [1.0], 0.0, 2, 0.0, 0.5, True

        liteopt_mod.gn = gn
        monkeypatch.setitem(sys.modules, "liteopt", liteopt_mod)

        runtime, _x_var = _build_scalar_runtime(target=1.0)
        iters: list[tuple[int, float, float]] = []

        def on_iter(k: int, rnorm: float, dxnorm: float) -> None:
            iters.append((int(k), float(rnorm), float(dxnorm)))

        out = solve(
            runtime,
            solver="liteopt",
            on_iter=on_iter,
            options={"method": "gn", "max_iters": 20},
        )

        assert out.converged
        assert len(iters) >= 2
        assert iters[0][0] == 0

    def test_solve_liteopt_retries_with_smaller_step_when_nonfinite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        step_calls: list[float] = []
        liteopt_mod = types.ModuleType("liteopt")

        def gd(fun, grad, x0, *, step_size=1e-3, max_iters=200, tol_grad=1e-4):
            step_calls.append(float(step_size))
            if float(step_size) > 0.01:
                raise ValueError("objective function must return finite float")

            x = np.asarray(x0, dtype=float).reshape(-1).copy()
            for k in range(int(max_iters)):
                g = np.asarray(grad(x), dtype=float).reshape(-1)
                if float(np.linalg.norm(g)) < float(tol_grad):
                    return x.copy(), float(fun(x)), True, int(k)
                x = x - float(step_size) * g
            return x.copy(), float(fun(x)), False, int(max_iters)

        liteopt_mod.gd = gd
        monkeypatch.setitem(sys.modules, "liteopt", liteopt_mod)

        runtime, _x_var = _build_scalar_runtime(target=3.0)
        out = solve(
            runtime,
            solver="liteopt",
            options={
                "max_iters": 2000,
                "step_size": 0.1,
                "tol_grad": 1e-6,
                "nonfinite_retries": 4,
                "nonfinite_step_shrink": 0.2,
            },
        )

        assert out.converged
        assert step_calls == pytest.approx([0.1, 0.02, 0.004], rel=0.0, abs=1e-12)
        assert out.meta["retry_count"] == 2
        assert float(out.meta["step_size_used"]) == pytest.approx(0.004, rel=0.0, abs=1e-12)

    def test_solve_liteopt_returns_failed_outcome_when_nonfinite_persists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[float] = []
        liteopt_mod = types.ModuleType("liteopt")

        def gd(fun, grad, x0, *, step_size=1e-3, max_iters=200, tol_grad=1e-4):
            del fun, grad, x0, max_iters, tol_grad
            calls.append(float(step_size))
            raise ValueError("objective function must return finite float")

        liteopt_mod.gd = gd
        monkeypatch.setitem(sys.modules, "liteopt", liteopt_mod)

        runtime, _x_var = _build_scalar_runtime(target=3.0)
        out = solve(
            runtime,
            solver="liteopt",
            options={
                "step_size": 0.1,
                "max_iters": 20,
                "nonfinite_retries": 2,
                "nonfinite_step_shrink": 0.1,
            },
        )

        assert not out.converged
        assert out.status == "failed"
        assert "non-finite" in str(out.stats.message).lower()
        assert calls == pytest.approx([0.1, 0.01, 0.001], rel=0.0, abs=1e-12)

    def test_solve_liteopt_rejects_unknown_method(self, monkeypatch: pytest.MonkeyPatch) -> None:
        liteopt_mod = types.ModuleType("liteopt")

        def gd(fun, grad, x0, *, step_size=1e-3, max_iters=200, tol_grad=1e-4):
            del fun, grad, x0, step_size, max_iters, tol_grad
            return [0.0], 0.0, True

        liteopt_mod.gd = gd
        monkeypatch.setitem(sys.modules, "liteopt", liteopt_mod)

        runtime, _x_var = _build_scalar_runtime(target=0.0)
        with pytest.raises(ValueError, match="method must be 'gd' or 'gn'"):
            _ = solve(
                runtime,
                solver="liteopt",
                options={"method": "unknown"},
            )
