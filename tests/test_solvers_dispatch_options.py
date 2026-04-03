from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from rei.optimize.builder import compile_nls_problem
from rei.optimize.solvers import solve


def _build_scalar_runtime(target: float = 0.0):
    dsl = {
        "variables": [{"name": "x", "dim": 1, "init": [0.0]}],
        "terms": [
            {
                "expr": {
                    "type": "sub",
                    "a": {"type": "get_var", "var": "x"},
                    "b": {"type": "const", "var": "x", "value": [float(target)]},
                },
                "cost": {"type": "l2"},
            }
        ],
    }
    return compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})


class TestSolverDispatchOptions:
    def test_solve_gauss_newton_accepts_x0_via_options(self) -> None:
        runtime = _build_scalar_runtime(target=2.0)
        out = solve(
            runtime,
            solver="gauss_newton",
            options={
                "x0": [1.0],
                "max_iters": 8,
                "tol_r": 1e-14,
                "tol_dx": 1e-14,
            },
        )

        assert out.converged
        assert float(out.stats.initial_objective or 0.0) == pytest.approx(1.0, rel=0.0, abs=1e-14)
        assert np.allclose(np.asarray(out.meta["x0"], dtype=float), np.array([1.0], dtype=float))

    def test_solve_rejects_duplicate_x0_sources(self) -> None:
        runtime = _build_scalar_runtime(target=0.0)
        with pytest.raises(ValueError, match="either as keyword argument or options\\['x0'\\]"):
            _ = solve(
                runtime,
                solver="gauss_newton",
                x0=np.array([1.0], dtype=float),
                options={"x0": [0.0]},
            )

    def test_solve_cyipopt_forwards_unknown_top_level_options_to_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, object] = {}
        cyipopt_mod = types.ModuleType("cyipopt")

        def minimize_ipopt(*, fun, x0, jac, bounds=None, options=None, tol=None, callback=None):
            del callback
            calls["bounds"] = bounds
            calls["tol"] = tol
            calls["options"] = {} if options is None else dict(options)
            x = np.asarray(x0, dtype=float).reshape(-1).copy()
            _ = float(fun(x))
            _ = np.asarray(jac(x), dtype=float).reshape(-1)
            return types.SimpleNamespace(
                x=x.copy(),
                success=True,
                nit=3,
                message="ok",
            )

        cyipopt_mod.minimize_ipopt = minimize_ipopt
        monkeypatch.setitem(sys.modules, "cyipopt", cyipopt_mod)

        runtime = _build_scalar_runtime(target=0.0)
        out = solve(
            runtime,
            solver="cyipopt",
            options={
                "max_iters": 7,
                "tol": 1e-9,
                "print_level": 0,
                "acceptable_tol": 1e-8,
            },
        )

        assert out.converged
        assert calls["tol"] == pytest.approx(1e-9, rel=0.0, abs=0.0)
        assert calls["options"] == {
            "max_iter": 7,
            "print_level": 0,
            "acceptable_tol": 1e-8,
        }

    def test_solve_cyipopt_merges_explicit_backend_options_and_flat_passthrough(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, object] = {}
        cyipopt_mod = types.ModuleType("cyipopt")

        def minimize_ipopt(*, fun, x0, jac, bounds=None, options=None, tol=None, callback=None):
            del bounds, tol, callback
            calls["options"] = {} if options is None else dict(options)
            x = np.asarray(x0, dtype=float).reshape(-1).copy()
            _ = float(fun(x))
            _ = np.asarray(jac(x), dtype=float).reshape(-1)
            return types.SimpleNamespace(x=x.copy(), success=True, nit=0, message="")

        cyipopt_mod.minimize_ipopt = minimize_ipopt
        monkeypatch.setitem(sys.modules, "cyipopt", cyipopt_mod)

        runtime = _build_scalar_runtime(target=0.0)
        out = solve(
            runtime,
            solver="cyipopt",
            options={
                "max_iters": 8,
                "linear_solver": "mumps",
                "backend_options": {"print_level": 3, "linear_solver": "ma57"},
            },
        )

        assert out.converged
        assert calls["options"] == {
            "print_level": 3,
            "linear_solver": "ma57",
            "max_iter": 8,
        }

    def test_solve_cyipopt_rejects_other_solver_option_keys(self) -> None:
        runtime = _build_scalar_runtime(target=0.0)
        with pytest.raises(ValueError, match="tol_dx"):
            _ = solve(
                runtime,
                solver="cyipopt",
                options={"tol_dx": 1e-8},
            )

    def test_solve_gauss_newton_rejects_unknown_options(self) -> None:
        runtime = _build_scalar_runtime(target=0.0)
        with pytest.raises(ValueError, match="print_level"):
            _ = solve(
                runtime,
                solver="gauss_newton",
                options={"max_iters": 5, "print_level": 0},
            )
