from __future__ import annotations

import unittest

import numpy as np

from eiopt.core.bspline import (
    bspline_basis_derivative_matrices,
    bspline_basis_derivative_matrix,
    default_clamped_uniform_knots,
)
from eiopt.core.state_cache import OwnerKey, StateCache, StateKey
from eiopt.core.state_schema import DTYPE_DYNAMICS, DTYPE_KINEMATICS, DYNAMICS_FIELDS, canonical_field_name, jac_field
from eiopt.core.time_grid import TimeGrid
from eiopt.core.trajectory import TrajectoryMap
from eiopt.dsl.trajectory import (
    build_trajectory_map,
    build_trajectory_map_with_derivative,
    build_trajectory_maps_with_derivatives,
)
from eiopt import (
    build_term_gradient_matrix,
    compile_problem,
    estimate_weights_simplex,
    format_solve_report,
    get_named_expr_value,
)
from eiopt.backends._template import BackendDispatchStateBuilder
from eiopt.expr.nodes import GetStateExpr, GetVarExpr
from eiopt.solvers import solve_gauss_newton, solve_runtime
from eiopt.model import Problem, ProblemRuntime, DirectVectorExpr, RuntimeContext, L2Cost, Variable, VariablePack


class TestEiOptBasic(unittest.TestCase):
    def test_gauss_newton_solves_linear_scalar(self) -> None:
        x_var = Variable(name="x", x=np.array([0.0], dtype=float))
        pack = VariablePack([x_var])

        def value(ctx: RuntimeContext) -> np.ndarray:
            x = float(ctx.pack.vars[0].x[0])
            return np.array([x - 3.0], dtype=float)

        def blocks(ctx: RuntimeContext):
            return [np.array([[1.0]], dtype=float)]

        expr = DirectVectorExpr(name="x_minus_3", vars=[x_var], fn_value=value, fn_blocks=blocks)
        problem = Problem(variables=pack, terms=[(expr, L2Cost())])
        runtime = ProblemRuntime(problem=problem, ctx=RuntimeContext(pack=pack), required=[])

        x0 = pack.get().copy()
        x_star, cost, _iters, _rnorm, _dxnorm, converged = solve_gauss_newton(runtime, max_iters=5, tol_r=1e-14, tol_dx=1e-14)
        self.assertTrue(converged)
        self.assertLess(cost, 1e-20)
        self.assertAlmostEqual(float(x_star[0]), 3.0, places=10)
        self.assertAlmostEqual(float(x_var.x[0]), 3.0, places=10)
        report = format_solve_report(runtime, x0=x0, x_star=x_star)
        self.assertIn("x_minus_3", report)
        self.assertIn("Variables:", report)
        self.assertIn("x0=", report)

    def test_solve_runtime_dispatches_gauss_newton(self) -> None:
        x_var = Variable(name="x", x=np.array([0.0], dtype=float))
        pack = VariablePack([x_var])

        def value(ctx: RuntimeContext) -> np.ndarray:
            x = float(ctx.pack.vars[0].x[0])
            return np.array([x - 2.5], dtype=float)

        def blocks(ctx: RuntimeContext):
            return [np.array([[1.0]], dtype=float)]

        expr = DirectVectorExpr(name="x_minus_2_5", vars=[x_var], fn_value=value, fn_blocks=blocks)
        problem = Problem(variables=pack, terms=[(expr, L2Cost())])
        runtime = ProblemRuntime(problem=problem, ctx=RuntimeContext(pack=pack), required=[])

        x_star, cost, _iters, _rnorm, _dxnorm, converged = solve_runtime(
            runtime,
            solver="gauss_newton",
            max_iters=5,
            tol_r=1e-14,
            tol_dx=1e-14,
        )
        self.assertTrue(converged)
        self.assertLess(cost, 1e-20)
        self.assertAlmostEqual(float(x_star[0]), 2.5, places=10)

    def test_solve_runtime_raises_for_unknown_solver(self) -> None:
        x_var = Variable(name="x", x=np.array([0.0], dtype=float))
        pack = VariablePack([x_var])

        def value(ctx: RuntimeContext) -> np.ndarray:
            return np.array([float(ctx.pack.vars[0].x[0])], dtype=float)

        def blocks(ctx: RuntimeContext):
            return [np.array([[1.0]], dtype=float)]

        expr = DirectVectorExpr(name="x_identity", vars=[x_var], fn_value=value, fn_blocks=blocks)
        problem = Problem(variables=pack, terms=[(expr, L2Cost())])
        runtime = ProblemRuntime(problem=problem, ctx=RuntimeContext(pack=pack), required=[])

        with self.assertRaisesRegex(ValueError, "Unknown solver"):
            _ = solve_runtime(runtime, solver="unknown_solver")

    def test_runtime_set_cost_weight_updates_specific_term_and_invalidates_cache(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [2.0]}],
            "terms": [
                {
                    "expr": {"type": "get_var", "name": "keep_term", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 1.0},
                },
                {
                    "expr": {"type": "get_var", "name": "tune_term", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 4.0},
                },
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        r0, _J0 = runtime.linearize()
        self.assertTrue(np.allclose(r0, np.array([2.0, 4.0], dtype=float)))

        idx = runtime.set_cost_weight("tune_term", 9.0)
        self.assertEqual(idx, 1)

        r1, _J1 = runtime.linearize()
        self.assertTrue(np.allclose(r1, np.array([2.0, 6.0], dtype=float)))

    def test_runtime_set_cost_weight_rejects_ambiguous_term_name(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [1.0]}],
            "terms": [
                {
                    "expr": {"type": "get_var", "name": "dup", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 1.0},
                },
                {
                    "expr": {"type": "get_var", "name": "dup", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 2.0},
                },
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        with self.assertRaisesRegex(ValueError, "multiple terms matched"):
            runtime.set_cost_weight("dup", 3.0)

    def test_runtime_set_cost_weight_rejects_unweighted_cost(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [1.0]}],
            "terms": [
                {
                    "expr": {"type": "get_var", "name": "plain", "var": "x"},
                    "cost": {"type": "l2"},
                }
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        with self.assertRaisesRegex(TypeError, "does not support runtime weight updates"):
            runtime.set_cost_weight("plain", 3.0)

    def test_runtime_collects_term_attrs_and_finds_indices(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [2.0]}],
            "terms": [
                {
                    "is_constraint": True,
                    "expr": {"type": "get_var", "name": "constraint_term", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 1.0},
                },
                {
                    "attrs": {"group": "constraint", "phase": "path"},
                    "expr": {"type": "get_var", "name": "grouped_constraint_term", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 4.0},
                },
                {
                    "expr": {"type": "get_var", "name": "plain_term", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 1.0},
                },
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        self.assertEqual(runtime.find_term_indices(attr="is_constraint", value=True), [0])
        self.assertEqual(runtime.find_term_indices(attr="group", value="constraint"), [1])
        self.assertEqual(runtime.term_attrs("constraint_term").get("is_constraint"), True)
        self.assertEqual(runtime.term_attrs("grouped_constraint_term").get("phase"), "path")
        self.assertEqual(runtime.term_attrs("plain_term"), {})

    def test_runtime_set_cost_weight_by_attr_updates_all_matches(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [2.0]}],
            "terms": [
                {
                    "attrs": {"group": "constraint"},
                    "expr": {"type": "get_var", "name": "constraint_a", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 1.0},
                },
                {
                    "attrs": {"group": "constraint"},
                    "expr": {"type": "get_var", "name": "constraint_b", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 4.0},
                },
                {
                    "attrs": {"group": "objective"},
                    "expr": {"type": "get_var", "name": "objective", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 1.0},
                },
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        r0, _J0 = runtime.linearize()
        self.assertTrue(np.allclose(r0, np.array([2.0, 4.0, 2.0], dtype=float)))

        idxs = runtime.set_cost_weight_by_attr(attr="group", value="constraint", w=9.0)
        self.assertEqual(idxs, [0, 1])

        r1, _J1 = runtime.linearize()
        self.assertTrue(np.allclose(r1, np.array([6.0, 6.0, 2.0], dtype=float)))

    def test_runtime_set_cost_weight_by_attr_rejects_no_match(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [1.0]}],
            "terms": [
                {
                    "attrs": {"group": "objective"},
                    "expr": {"type": "get_var", "name": "plain", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 1.0},
                }
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        with self.assertRaisesRegex(ValueError, "no term matched"):
            runtime.set_cost_weight_by_attr(attr="group", value="constraint", w=3.0)

    def test_runtime_constraint_kind_helpers(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [2.0]}],
            "terms": [
                {
                    "constraint": {"kind": "eq"},
                    "expr": {"type": "get_var", "name": "eq_term", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 1.0},
                },
                {
                    "constraint": {"type": "ineq"},
                    "expr": {"type": "get_var", "name": "ineq_term", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 4.0},
                },
                {
                    "expr": {"type": "get_var", "name": "objective", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 1.0},
                },
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        self.assertEqual(runtime.find_constraint_term_indices(), [0, 1])
        self.assertEqual(runtime.find_constraint_term_indices(kind="eq"), [0])
        self.assertEqual(runtime.find_constraint_term_indices(kind="ineq"), [1])
        self.assertEqual(runtime.term_attrs("eq_term").get("constraint_kind"), "eq")
        self.assertEqual(runtime.term_attrs("ineq_term").get("constraint_kind"), "ineq")

    def test_runtime_set_cost_weight_by_constraint(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [2.0]}],
            "terms": [
                {
                    "constraint": "eq",
                    "expr": {"type": "get_var", "name": "eq_term", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 1.0},
                },
                {
                    "constraint": "ineq",
                    "expr": {"type": "get_var", "name": "ineq_term", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 4.0},
                },
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        idxs = runtime.set_cost_weight_by_constraint(kind="ineq", w=9.0)
        self.assertEqual(idxs, [1])

        r1, _J1 = runtime.linearize()
        self.assertTrue(np.allclose(r1, np.array([2.0, 6.0], dtype=float)))

    def test_runtime_linearize_terms_weighted_and_unweighted(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [2.0]}],
            "terms": [
                {
                    "attrs": {"group": "objective"},
                    "expr": {"type": "get_var", "name": "term_a", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 1.0},
                },
                {
                    "attrs": {"group": "constraint"},
                    "expr": {"type": "get_var", "name": "term_b", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 9.0},
                },
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        terms_raw = runtime.linearize_terms(weighted=False)
        terms_w = runtime.linearize_terms(weighted=True)

        self.assertEqual([t.term_index for t in terms_raw], [0, 1])
        self.assertEqual([t.name for t in terms_raw], ["term_a", "term_b"])
        self.assertEqual(terms_raw[1].attrs.get("group"), "constraint")

        self.assertTrue(np.allclose(terms_raw[0].residual, np.array([2.0], dtype=float)))
        self.assertTrue(np.allclose(terms_raw[1].residual, np.array([2.0], dtype=float)))
        self.assertTrue(np.allclose(terms_raw[0].jacobian, np.array([[1.0]], dtype=float)))
        self.assertTrue(np.allclose(terms_raw[1].jacobian, np.array([[1.0]], dtype=float)))

        self.assertTrue(np.allclose(terms_w[0].residual, np.array([2.0], dtype=float)))
        self.assertTrue(np.allclose(terms_w[1].residual, np.array([6.0], dtype=float)))
        self.assertTrue(np.allclose(terms_w[0].jacobian, np.array([[1.0]], dtype=float)))
        self.assertTrue(np.allclose(terms_w[1].jacobian, np.array([[3.0]], dtype=float)))

    def test_runtime_linearize_constraint_terms(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [2.0]}],
            "terms": [
                {
                    "constraint": "eq",
                    "expr": {"type": "get_var", "name": "eq_term", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 1.0},
                },
                {
                    "constraint": "ineq",
                    "expr": {"type": "get_var", "name": "ineq_term", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 4.0},
                },
                {
                    "expr": {"type": "get_var", "name": "objective_term", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 1.0},
                },
            ],
        }
        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})

        eq_terms = runtime.linearize_constraint_terms(kind="eq", weighted=False)
        ineq_terms = runtime.linearize_constraint_terms(kind="ineq", weighted=False)
        self.assertEqual([t.name for t in eq_terms], ["eq_term"])
        self.assertEqual([t.name for t in ineq_terms], ["ineq_term"])
        self.assertTrue(np.allclose(eq_terms[0].residual, np.array([2.0], dtype=float)))
        self.assertTrue(np.allclose(ineq_terms[0].residual, np.array([2.0], dtype=float)))

    def test_runtime_collect_state_traj_helper(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [0.0]}],
            "terms": [
                {
                    "expr": {"type": "get_var", "name": "identity", "var": "x"},
                    "cost": {"type": "l2"},
                }
            ],
        }

        owner = OwnerKey(owner_type="demo", owner_name="thing")

        def build_state(_x_all: np.ndarray, *, required=None, **_kwargs) -> dict[StateKey, object]:
            req = [] if required is None else list(required)
            out: dict[StateKey, object] = {}
            for key in req:
                if key.owner != owner or key.dtype != "vec" or key.field != "y":
                    continue
                out[key] = np.array([float(key.k), float(key.k) + 10.0], dtype=float)
            return out

        runtime = compile_problem(dsl, build_state=build_state)
        traj = runtime.collect_state_traj(
            owner_type="demo",
            owner_name="thing",
            dtype="vec",
            field="y",
            ks=[0, 1, 2],
            expected_dim=2,
        )
        self.assertEqual(traj.shape, (3, 2))
        self.assertTrue(np.allclose(traj[0], np.array([0.0, 10.0], dtype=float)))
        self.assertTrue(np.allclose(traj[2], np.array([2.0, 12.0], dtype=float)))

    def test_ioc_matrix_and_simplex_weight_estimation(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "name": "t1",
                        "a": {"type": "get_var", "var": "x"},
                        "b": {"type": "const", "var": "x", "value": [1.0]},
                    },
                    "cost": {"type": "l2"},
                },
                {
                    "expr": {
                        "type": "sub",
                        "name": "t2",
                        "a": {"type": "get_var", "var": "x"},
                        "b": {"type": "const", "var": "x", "value": [-1.0]},
                    },
                    "cost": {"type": "l2"},
                },
            ],
        }
        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})

        A, term_indices = build_term_gradient_matrix(runtime, weighted=False)
        self.assertEqual(A.shape, (1, 2))
        self.assertEqual(term_indices, [0, 1])
        self.assertTrue(np.allclose(A, np.array([[-1.0, 1.0]], dtype=float)))

        w, info = estimate_weights_simplex(A, return_info=True)
        self.assertTrue(np.allclose(w, np.array([0.5, 0.5], dtype=float), atol=1e-6))
        self.assertAlmostEqual(float(np.sum(w)), 1.0, places=8)
        self.assertTrue(bool(info["converged"]))
        self.assertLess(float(info["objective"]), 1e-12)

    def test_format_solve_report_includes_diagnostics(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [1.5]}],
            "terms": [
                {
                    "expr": {"type": "get_var", "name": "term_x", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 4.0},
                }
            ],
        }
        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        report = format_solve_report(runtime)
        self.assertIn("Diagnostics:", report)
        self.assertIn("||J^T r||", report)
        self.assertIn("rank(J)", report)
        self.assertIn("svd(J)", report)
        self.assertIn("active terms", report)

    def test_constraint_kind_validation(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [1.0]}],
            "terms": [
                {
                    "constraint": {"kind": "not_supported"},
                    "expr": {"type": "get_var", "name": "bad", "var": "x"},
                    "cost": {"type": "scalar_weight", "w": 1.0},
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "must be 'eq' or 'ineq'"):
            _ = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})

    def test_get_state_expr_reads_cache(self) -> None:
        q_var = Variable(name="q", x=np.array([1.0, 2.0], dtype=float))
        pack = VariablePack([q_var])
        time = TimeGrid.single_time()

        owner = OwnerKey(owner_type="demo", owner_name="thing")
        key_val = StateKey(k=0, owner=owner, dtype="vec", field="y")
        key_jac = StateKey(k=0, owner=owner, dtype="vec", field="y_J_q")

        def build_state(
            x_all: np.ndarray, *, pack=None, time=None, required=None
        ) -> dict[StateKey, object]:
            q = np.asarray(x_all, dtype=float).reshape(-1)
            out: dict[StateKey, object] = {}
            if required is None:
                required = [key_val, key_jac]
            required_set = set(required)
            if key_val in required_set:
                out[key_val] = q.copy()
            if key_jac in required_set:
                out[key_jac] = np.eye(q.size, dtype=float)
            return out

        cache = StateCache(build_state=build_state)
        cache.update_if_needed(pack, time=time, required=[key_val, key_jac])

        expr = GetStateExpr(name="get_y", vars=[q_var], key_value=key_val, key_jacs=[key_jac])
        ctx = RuntimeContext(pack=pack, state=cache, time=time)

        y, blocks = expr.eval(ctx)
        self.assertTrue(np.allclose(y, np.array([1.0, 2.0])))
        self.assertEqual(len(blocks), 1)
        self.assertTrue(np.allclose(blocks[0], np.eye(2)))

    def test_get_state_builder_autofills_jac_field(self) -> None:
        dsl = {
            "variables": [{"name": "q", "dim": 2, "init": [0.0, 0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "key": {
                            "k": 0,
                            "owner_type": "link",
                            "owner_name": "end",
                            "dtype": "kinematics",
                            "field": "pos",
                        },
                        "jac": {"var": "q"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        def build_state(_x_all: np.ndarray, *, pack=None, time=None, required=None):
            return {}

        runtime = compile_problem(dsl, build_state=build_state)
        fields = {k.field for k in runtime.required}
        self.assertIn("pos", fields)
        self.assertIn(jac_field("pos", var="q"), fields)
        frames = {k.frame for k in runtime.required}
        self.assertEqual(frames, {"world"})

    def test_get_state_expr_multiple_jac_blocks(self) -> None:
        p_var = Variable(name="p", x=np.array([0.1, 0.2], dtype=float))
        q_var = Variable(name="q", x=np.array([1.0, 2.0, 3.0], dtype=float))
        pack = VariablePack([p_var, q_var])
        time = TimeGrid.single_time()

        owner = OwnerKey(owner_type="demo", owner_name="thing")
        key_val = StateKey(k=0, owner=owner, dtype="vec", field="y")
        key_jac_p = StateKey(k=0, owner=owner, dtype="vec", field="y_J_p")
        key_jac_q = StateKey(k=0, owner=owner, dtype="vec", field="y_J_q")

        def build_state(_x_all: np.ndarray, *, required=None, **_kwargs) -> dict[StateKey, object]:
            req = set(required) if required is not None else {key_val, key_jac_p, key_jac_q}
            out: dict[StateKey, object] = {}
            if key_val in req:
                out[key_val] = np.array([2.0, -1.0], dtype=float)
            if key_jac_p in req:
                out[key_jac_p] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
            if key_jac_q in req:
                out[key_jac_q] = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=float)
            return out

        cache = StateCache(build_state=build_state)
        cache.update_if_needed(pack, time=time, required=[key_val, key_jac_p, key_jac_q])

        expr = GetStateExpr(
            name="get_y_multi",
            vars=[p_var, q_var],
            key_value=key_val,
            key_jacs=[key_jac_p, key_jac_q],
        )
        y, blocks = expr.eval(RuntimeContext(pack=pack, state=cache, time=time))

        self.assertTrue(np.allclose(y, np.array([2.0, -1.0], dtype=float)))
        self.assertEqual(len(blocks), 2)
        self.assertTrue(np.allclose(blocks[0], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)))
        self.assertTrue(np.allclose(blocks[1], np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=float)))

    def test_get_state_builder_supports_jacs_list(self) -> None:
        dsl = {
            "variables": [
                {"name": "p", "dim": 2, "init": [0.0, 0.0]},
                {"name": "q", "dim": 2, "init": [0.0, 0.0]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "key": {
                            "k": 0,
                            "owner_type": "link",
                            "owner_name": "end",
                            "dtype": "kinematics",
                            "field": "pos",
                        },
                        "jacs": [
                            {"var": "p"},
                            {"var": "q"},
                        ],
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        fields = {k.field for k in runtime.required}
        self.assertIn("pos", fields)
        self.assertIn(jac_field("pos", var="p"), fields)
        self.assertIn(jac_field("pos", var="q"), fields)

    def test_get_state_builder_requires_explicit_jac_var_when_multiple_vars(self) -> None:
        dsl = {
            "variables": [
                {"name": "p", "dim": 2, "init": [0.0, 0.0]},
                {"name": "q", "dim": 2, "init": [0.0, 0.0]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "key": {
                            "k": 0,
                            "owner_type": "link",
                            "owner_name": "end",
                            "dtype": "kinematics",
                            "field": "pos",
                        },
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "Multiple variables exist"):
            _ = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})

    def test_get_state_builder_canonicalizes_tau_alias(self) -> None:
        dsl = {
            "variables": [{"name": "q", "dim": 2, "init": [0.0, 0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "key": {
                            "k": 0,
                            "owner_type": "total_joint",
                            "owner_name": "robot",
                            "dtype": DTYPE_DYNAMICS,
                            "field": "tau",
                        },
                        "jac": {"var": "q"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        fields = {k.field for k in runtime.required}
        self.assertIn("torque", fields)
        self.assertIn("torque_J_q", fields)

    def test_get_state_builder_canonicalizes_dtau_alias(self) -> None:
        dsl = {
            "variables": [{"name": "q", "dim": 2, "init": [0.0, 0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "key": {
                            "k": 0,
                            "owner_type": "total_joint",
                            "owner_name": "robot",
                            "dtype": DTYPE_DYNAMICS,
                            "field": "dtau",
                        },
                        "jac": {"var": "q"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        fields = {k.field for k in runtime.required}
        self.assertIn("torque_rate", fields)
        self.assertIn("torque_rate_J_q", fields)

    def test_state_schema_dynamics_field_aliases(self) -> None:
        self.assertIn("torque", DYNAMICS_FIELDS)
        self.assertIn("torque_rate", DYNAMICS_FIELDS)
        self.assertEqual(canonical_field_name("tau"), "torque")
        self.assertEqual(canonical_field_name("dtau"), "torque_rate")
        self.assertEqual(canonical_field_name("h"), "momentum")
        self.assertEqual(canonical_field_name("wrench"), "force")
        self.assertEqual(canonical_field_name("dtau_J_p"), "torque_rate_J_p")

    def test_stack_get_state_canonicalizes_dtau_alias(self) -> None:
        dsl = {
            "time": {"N": 1, "dt": 0.1},
            "variables": [{"name": "p", "dim": 2, "init": [0.0, 0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "stack",
                        "range": {"k0": 0, "k1": 1},
                        "inner": {
                            "type": "get_state",
                            "key": {
                                "owner_type": "total_joint",
                                "owner_name": "robot",
                                "dtype": DTYPE_DYNAMICS,
                                "field": "dtau",
                            },
                            "jac": {"var": "p"},
                        },
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        torque_rate_keys = [k for k in runtime.required if k.field == "torque_rate"]
        torque_rate_jac_keys = [k for k in runtime.required if k.field == "torque_rate_J_p"]
        self.assertEqual({k.k for k in torque_rate_keys}, {0, 1})
        self.assertEqual({k.k for k in torque_rate_jac_keys}, {0, 1})

    def test_get_var_expr_reads_pack(self) -> None:
        q_var = Variable(name="q", x=np.array([1.0, 2.0], dtype=float))
        pack = VariablePack([q_var])
        time = TimeGrid.single_time()

        expr = GetVarExpr(name="get_q", vars=[q_var])
        ctx = RuntimeContext(pack=pack, state=None, time=time)

        q, blocks = expr.eval(ctx)
        self.assertTrue(np.allclose(q, np.array([1.0, 2.0], dtype=float)))
        self.assertEqual(len(blocks), 1)
        self.assertTrue(np.allclose(blocks[0], np.eye(2, dtype=float)))

    def test_get_var_expr_slices_time_chunked(self) -> None:
        q_var = Variable(name="q", x=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float))
        pack = VariablePack([q_var])
        time = TimeGrid(N=2, dt=0.1)  # k=0,1,2

        expr = GetVarExpr(name="get_q1", vars=[q_var], k=1)
        ctx = RuntimeContext(pack=pack, state=None, time=time)

        q1, blocks = expr.eval(ctx)
        J1 = np.asarray(blocks[0], dtype=float)

        self.assertTrue(np.allclose(q1, np.array([3.0, 4.0], dtype=float)))

        expected_J = np.zeros((2, 6), dtype=float)
        expected_J[:, 2:4] = np.eye(2, dtype=float)
        self.assertTrue(np.allclose(J1, expected_J))

    def test_get_named_expr_value_single(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 2, "init": [1.5, -2.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "name": "x_err",
                        "a": {"type": "get_var", "name": "x_now", "var": "x"},
                        "b": {"type": "const", "name": "x_ref", "var": "x", "value": [0.0, 0.0]},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        x_now = get_named_expr_value(runtime, name="x_now")
        self.assertTrue(np.allclose(x_now, np.array([1.5, -2.0], dtype=float)))

    def test_get_named_expr_value_raises_on_ambiguous_match(self) -> None:
        dsl = {
            "time": {"N": 1, "dt": 0.1},
            "variables": [{"name": "x", "dim": 2, "init": [1.0, 2.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "stack",
                        "name": "x_stack",
                        "range": {"k0": 0, "k1": 1},
                        "inner": {"type": "get_var", "name": "x_k", "var": "x"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        with self.assertRaisesRegex(ValueError, "multiple named Expr values matched"):
            _ = get_named_expr_value(runtime, name="x_k")

    def test_get_traj_var_expr_bspline_inferrs_q_dim_from_var(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.1},
            "trajectory": {
                "type": "bspline",
                "var": "p",
                "degree": 1,
                "num_ctrl_points": 2,
            },
            "variables": [
                {"name": "p", "dim": 4, "init": [1.0, 2.0, 3.0, 4.0]},
            ],
            "terms": [
                {
                    "expr": {"type": "get_traj_var", "name": "q_traj", "var": "p"},
                    "cost": {"type": "l2"},
                }
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        r, J = runtime.linearize()

        traj = build_trajectory_map(
            dsl["trajectory"],
            default_steps=3,
            default_q_dim=2,
        )
        p = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        self.assertTrue(np.allclose(r, traj.A @ p + traj.b))
        self.assertTrue(np.allclose(J, traj.A))

    def test_get_traj_var_regularization_term_against_previous_trajectory(self) -> None:
        prev_q = np.array(
            [0.5, 1.5, 1.0, 2.0, 1.5, 2.5],
            dtype=float,
        )
        dsl = {
            "time": {"N": 2, "dt": 0.1},
            "trajectory": {
                "type": "bspline",
                "var": "p",
                "degree": 1,
                "num_ctrl_points": 2,
            },
            "variables": [
                {"name": "p", "dim": 4, "init": [1.0, 2.0, 3.0, 4.0]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "name": "traj_prev_diff",
                        "a": {"type": "get_traj_var", "name": "q_traj", "var": "p"},
                        "b": {"type": "const", "name": "q_prev", "var": "p", "value": prev_q.tolist()},
                    },
                    "cost": {"type": "scalar_weight", "w": 4.0},
                }
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        r, J = runtime.linearize()

        traj = build_trajectory_map(
            dsl["trajectory"],
            default_steps=3,
            default_q_dim=2,
        )
        p = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        q = traj.A @ p + traj.b
        self.assertTrue(np.allclose(r, 2.0 * (q - prev_q)))
        self.assertTrue(np.allclose(J, 2.0 * traj.A))

    def test_get_traj_var_expr_bspline_first_derivative_wrt_time(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.2},
            "trajectory": {
                "type": "bspline",
                "var": "p",
                "degree": 1,
                "num_ctrl_points": 2,
                "q_dim": 2,
            },
            "variables": [
                {"name": "p", "dim": 4, "init": [1.0, 2.0, 3.0, 4.0]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "get_traj_var",
                        "name": "qdot_traj",
                        "var": "p",
                        "derivative_order": 1,
                        "derivative_wrt": "time",
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        r, J = runtime.linearize()

        traj_d1 = build_trajectory_map_with_derivative(
            dsl["trajectory"],
            derivative_order=1,
            derivative_wrt="time",
            default_steps=3,
            default_q_dim=2,
            default_dt=0.2,
        )
        p = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        self.assertTrue(np.allclose(r, traj_d1.A @ p + traj_d1.b))
        self.assertTrue(np.allclose(J, traj_d1.A))

        qdot = r.reshape(3, 2)
        self.assertTrue(np.allclose(qdot, np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]], dtype=float)))

    def test_get_traj_var_expr_bspline_derivatives_0_to_n(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.2},
            "trajectory": {
                "type": "bspline",
                "var": "p",
                "degree": 1,
                "num_ctrl_points": 2,
                "q_dim": 2,
            },
            "variables": [
                {"name": "p", "dim": 4, "init": [1.0, 2.0, 3.0, 4.0]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "get_traj_var",
                        "name": "q_and_qdot",
                        "var": "p",
                        "max_derivative_order": 1,
                        "derivative_wrt": "time",
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        r, J = runtime.linearize()

        maps = build_trajectory_maps_with_derivatives(
            dsl["trajectory"],
            max_derivative_order=1,
            derivative_wrt="time",
            default_steps=3,
            default_q_dim=2,
            default_dt=0.2,
        )
        p = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        r_ref = np.concatenate([m.A @ p + m.b for m in maps], axis=0)
        J_ref = np.vstack([m.A for m in maps])

        self.assertEqual(r.size, 12)
        self.assertEqual(J.shape, (12, 4))
        self.assertTrue(np.allclose(r, r_ref))
        self.assertTrue(np.allclose(J, J_ref))

    def test_get_traj_var_expr_bspline_derivatives_0_to_n_with_k_slice(self) -> None:
        dsl = {
            "time": {"N": 4, "dt": 0.1},
            "trajectory": {
                "type": "bspline",
                "var": "p",
                "degree": 2,
                "num_ctrl_points": 4,
                "q_dim": 2,
            },
            "variables": [
                {"name": "p", "dim": 8, "init": [0.2, -0.1, 0.4, 0.7, 1.0, -0.3, 0.1, 0.6]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "get_traj_var",
                        "name": "q_and_dq_and_ddq_k2",
                        "var": "p",
                        "max_derivative_order": 2,
                        "derivative_wrt": "u",
                        "k": 2,
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        r, J = runtime.linearize()

        maps = build_trajectory_maps_with_derivatives(
            dsl["trajectory"],
            max_derivative_order=2,
            derivative_wrt="u",
            default_steps=5,
            default_q_dim=2,
            default_dt=0.1,
        )
        p = np.array([0.2, -0.1, 0.4, 0.7, 1.0, -0.3, 0.1, 0.6], dtype=float)
        r_ref = np.concatenate([m.q_at(p, 2) for m in maps], axis=0)
        J_ref = np.vstack([m.dqdp_at(2) for m in maps])

        self.assertEqual(r.size, 6)
        self.assertEqual(J.shape, (6, 8))
        self.assertTrue(np.allclose(r, r_ref))
        self.assertTrue(np.allclose(J, J_ref))

    def test_bspline_basis_derivative_matrices_matches_single_order(self) -> None:
        degree = 4
        num_ctrl_points = 8
        knots = default_clamped_uniform_knots(
            num_ctrl_points=num_ctrl_points,
            degree=degree,
        )
        u_vec = np.linspace(float(knots[degree]), float(knots[num_ctrl_points]), 11, dtype=float)

        max_order = 6
        mats = bspline_basis_derivative_matrices(
            u_vec=u_vec,
            degree=degree,
            knots=knots,
            num_ctrl_points=num_ctrl_points,
            max_derivative_order=max_order,
        )
        self.assertEqual(mats.shape, (max_order + 1, u_vec.size, num_ctrl_points))

        for order in range(max_order + 1):
            mat_single = bspline_basis_derivative_matrix(
                u_vec=u_vec,
                degree=degree,
                knots=knots,
                num_ctrl_points=num_ctrl_points,
                derivative_order=order,
            )
            self.assertTrue(np.allclose(mats[order, :, :], mat_single, atol=1e-10, rtol=1e-10))

        self.assertTrue(np.allclose(mats[degree + 1 :, :, :], 0.0))

    def test_trajectory_map_from_bspline_derivatives_matches_single_order(self) -> None:
        steps = 7
        q_dim = 2
        degree = 3
        num_ctrl_points = 6
        parameter_scale = 1.7
        max_order = 5

        maps = TrajectoryMap.from_bspline_derivatives(
            steps=steps,
            q_dim=q_dim,
            degree=degree,
            num_ctrl_points=num_ctrl_points,
            max_derivative_order=max_order,
            parameter_scale=parameter_scale,
        )
        self.assertEqual(len(maps), max_order + 1)

        p = np.arange(num_ctrl_points * q_dim, dtype=float) * 0.1
        for order in range(max_order + 1):
            map_single = TrajectoryMap.from_bspline_derivative(
                steps=steps,
                q_dim=q_dim,
                degree=degree,
                num_ctrl_points=num_ctrl_points,
                derivative_order=order,
                parameter_scale=parameter_scale,
            )
            self.assertTrue(np.allclose(maps[order].A, map_single.A))
            self.assertTrue(np.allclose(maps[order].b, map_single.b))
            for k in range(steps):
                self.assertTrue(np.allclose(maps[order].q_at(p, k), map_single.q_at(p, k)))

    def test_build_trajectory_maps_with_derivatives_time_matches_single(self) -> None:
        traj_dsl = {
            "type": "bspline",
            "var": "p",
            "degree": 3,
            "num_ctrl_points": 6,
            "q_dim": 2,
            "steps": 8,
        }
        maps = build_trajectory_maps_with_derivatives(
            traj_dsl,
            max_derivative_order=4,
            derivative_wrt="time",
            default_dt=0.2,
        )
        self.assertEqual(len(maps), 5)

        map_d2 = build_trajectory_map_with_derivative(
            traj_dsl,
            derivative_order=2,
            derivative_wrt="time",
            default_dt=0.2,
        )
        self.assertTrue(np.allclose(maps[2].A, map_d2.A))
        self.assertTrue(np.allclose(maps[2].b, map_d2.b))

    def test_get_traj_var_expr_slices_by_k(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.1},
            "trajectory": {
                "type": "bspline",
                "var": "p",
                "degree": 1,
                "num_ctrl_points": 2,
                "q_dim": 2,
            },
            "variables": [
                {"name": "p", "dim": 4, "init": [1.0, 2.0, 3.0, 4.0]},
            ],
            "terms": [
                {
                    "expr": {"type": "get_traj_var", "name": "q_traj_k1", "var": "p", "k": 1},
                    "cost": {"type": "l2"},
                }
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        r, J = runtime.linearize()

        traj = build_trajectory_map(
            dsl["trajectory"],
            default_steps=3,
            default_q_dim=2,
        )
        p = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

        self.assertTrue(np.allclose(r, traj.q_at(p, 1)))
        self.assertTrue(np.allclose(J, traj.dqdp_at(1)))

    def test_time_diff_expr_on_traj_var(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.1},
            "trajectory": {
                "type": "bspline",
                "var": "p",
                "degree": 1,
                "num_ctrl_points": 2,
                "q_dim": 2,
            },
            "variables": [
                {"name": "p", "dim": 4, "init": [1.0, 2.0, 3.0, 4.0]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "time_diff",
                        "name": "traj_smooth",
                        "segment_dim": 2,
                        "base": {"type": "get_traj_var", "name": "q_traj", "var": "p"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        r, J = runtime.linearize()

        traj = build_trajectory_map(
            dsl["trajectory"],
            default_steps=3,
            default_q_dim=2,
        )
        p = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        q = (traj.A @ p + traj.b).reshape(3, 2)
        r_ref = (q[1:, :] - q[:-1, :]).reshape(-1)

        D = np.zeros((4, 6), dtype=float)
        D[0:2, 0:2] = -np.eye(2, dtype=float)
        D[0:2, 2:4] = np.eye(2, dtype=float)
        D[2:4, 2:4] = -np.eye(2, dtype=float)
        D[2:4, 4:6] = np.eye(2, dtype=float)
        J_ref = D @ traj.A

        self.assertTrue(np.allclose(r, r_ref))
        self.assertTrue(np.allclose(J, J_ref))

    def test_time_diff_expr_wrt_time_scales_by_dt(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.5},
            "variables": [
                {"name": "q", "dim": 6, "init": [0.0, 0.0, 1.0, 2.0, 3.0, 6.0]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "time_diff",
                        "name": "dqdt",
                        "segment_dim": 2,
                        "wrt": "time",
                        "base": {"type": "get_var", "var": "q"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        r, J = runtime.linearize()

        q = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 6.0], dtype=float).reshape(3, 2)
        r_ref = ((q[1:, :] - q[:-1, :]) / 0.5).reshape(-1)

        D = np.zeros((4, 6), dtype=float)
        D[0:2, 0:2] = -2.0 * np.eye(2, dtype=float)
        D[0:2, 2:4] = 2.0 * np.eye(2, dtype=float)
        D[2:4, 2:4] = -2.0 * np.eye(2, dtype=float)
        D[2:4, 4:6] = 2.0 * np.eye(2, dtype=float)

        self.assertTrue(np.allclose(r, r_ref))
        self.assertTrue(np.allclose(J, D))

    def test_const_repeat_expr_uses_time_steps(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.1},
            "variables": [{"name": "p", "dim": 2, "init": [0.0, 0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "const_repeat",
                        "name": "q_max_traj",
                        "var": "p",
                        "value": [1.0, -1.0],
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        runtime = compile_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        r, J = runtime.linearize()

        self.assertTrue(np.allclose(r, np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=float)))
        self.assertEqual(J.shape, (6, 2))
        self.assertTrue(np.allclose(J, 0.0))

    def test_build_trajectory_maps_with_derivatives_linear_uses_finite_difference(self) -> None:
        traj_dsl = {
            "type": "linear",
            "steps": 3,
            "q_dim": 1,
            "linear": {
                "A": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        }
        maps = build_trajectory_maps_with_derivatives(
            traj_dsl,
            max_derivative_order=1,
            derivative_wrt="time",
            default_dt=0.2,
        )
        self.assertEqual(len(maps), 2)
        self.assertEqual(maps[0].steps, 3)
        self.assertEqual(maps[1].steps, 3)

        p = np.array([0.0, 1.0, 3.0], dtype=float)
        q = (maps[0].A @ p + maps[0].b).reshape(3)
        dq = (maps[1].A @ p + maps[1].b).reshape(3)

        dq_ref = np.array(
            [
                (q[1] - q[0]) / 0.2,
                (q[2] - q[0]) / 0.4,
                (q[2] - q[1]) / 0.2,
            ],
            dtype=float,
        )
        self.assertTrue(np.allclose(dq, dq_ref))

    def test_build_trajectory_maps_with_derivatives_bspline_nonuniform_time_fallback(self) -> None:
        traj_dsl = {
            "type": "bspline",
            "steps": 3,
            "q_dim": 1,
            "degree": 1,
            "num_ctrl_points": 2,
            "u_samples": [0.0, 0.25, 1.0],
        }
        maps = build_trajectory_maps_with_derivatives(
            traj_dsl,
            max_derivative_order=1,
            derivative_wrt="time",
            default_dt=0.5,
        )
        self.assertEqual(len(maps), 2)

        p = np.array([1.0, 5.0], dtype=float)
        q = (maps[0].A @ p + maps[0].b).reshape(3)
        dq = (maps[1].A @ p + maps[1].b).reshape(3)
        dq_ref = np.array(
            [
                (q[1] - q[0]) / 0.5,
                (q[2] - q[0]) / 1.0,
                (q[2] - q[1]) / 0.5,
            ],
            dtype=float,
        )
        self.assertTrue(np.allclose(dq, dq_ref))

    def test_trajectory_map_q_and_jac(self) -> None:
        A = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.5, 0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0, 0.5],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        b = np.array([0.1, -0.2, 0.0, 0.0, -0.3, 0.4], dtype=float)
        traj = TrajectoryMap(A=A, b=b, steps=3, q_dim=2)
        p = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

        q0 = traj.q_at(p, 0)
        q1 = traj.q_at(p, 1)
        q2 = traj.q_at(p, 2)

        self.assertTrue(np.allclose(q0, np.array([1.1, 1.8], dtype=float)))
        self.assertTrue(np.allclose(q1, np.array([2.0, 3.0], dtype=float)))
        self.assertTrue(np.allclose(q2, np.array([2.7, 4.4], dtype=float)))

        J1 = traj.dqdp_at(1)
        self.assertTrue(np.allclose(J1, np.array([[0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5]], dtype=float)))

    def test_trajectory_map_from_blocks(self) -> None:
        A_blocks = [
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
            np.array([[0.25, 0.0], [0.0, 0.25]], dtype=float),
        ]
        b_blocks = [
            np.array([0.0, 0.0], dtype=float),
            np.array([1.0, -1.0], dtype=float),
        ]
        traj = TrajectoryMap.from_blocks(A_blocks, b_blocks=b_blocks)
        p = np.array([4.0, 8.0], dtype=float)

        self.assertEqual(traj.steps, 2)
        self.assertEqual(traj.q_dim, 2)
        self.assertEqual(traj.p_dim, 2)
        self.assertTrue(np.allclose(traj.q_at(p, 0), np.array([4.0, 8.0], dtype=float)))
        self.assertTrue(np.allclose(traj.q_at(p, 1), np.array([2.0, 1.0], dtype=float)))

    def test_bspline_trajectory_map_degree1_equals_linear_interp(self) -> None:
        steps = 5
        q_dim = 2
        traj = TrajectoryMap.from_bspline(
            steps=steps,
            q_dim=q_dim,
            degree=1,
            num_ctrl_points=2,
        )
        q_start = np.array([1.2, -0.3], dtype=float)
        q_goal = np.array([4.2, 2.7], dtype=float)
        p = np.concatenate([q_start, q_goal], axis=0)

        for k in range(steps):
            alpha = float(k) / float(steps - 1)
            q_ref = (1.0 - alpha) * q_start + alpha * q_goal
            self.assertTrue(np.allclose(traj.q_at(p, k), q_ref))

    def test_bspline_trajectory_map_jacobian_matches_finite_difference(self) -> None:
        traj = TrajectoryMap.from_bspline(
            steps=4,
            q_dim=3,
            degree=3,
            num_ctrl_points=5,
        )
        p = np.array(
            [
                -0.5,
                0.1,
                0.4,
                0.2,
                -0.3,
                0.7,
                1.0,
                -0.2,
                0.6,
                0.3,
                0.8,
                -0.4,
                -0.1,
                0.2,
                -0.6,
            ],
            dtype=float,
        )
        eps = 1e-7

        for k in range(traj.steps):
            J = traj.dqdp_at(k)
            self.assertEqual(J.shape, (traj.q_dim, traj.p_dim))
            q0 = traj.q_at(p, k)
            J_fd = np.zeros_like(J)
            for j in range(traj.p_dim):
                dp = np.zeros((traj.p_dim,), dtype=float)
                dp[j] = eps
                q1 = traj.q_at(p + dp, k)
                J_fd[:, j] = (q1 - q0) / eps
            self.assertTrue(np.allclose(J, J_fd, atol=1e-6, rtol=1e-6))

    def test_build_trajectory_map_bspline(self) -> None:
        dsl = {
            "type": "bspline",
            "var": "p",
            "degree": 1,
            "num_ctrl_points": 2,
        }
        traj = build_trajectory_map(dsl, default_steps=4, default_q_dim=2)

        self.assertEqual(traj.steps, 4)
        self.assertEqual(traj.q_dim, 2)
        self.assertEqual(traj.p_dim, 4)

        p = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        self.assertTrue(np.allclose(traj.q_at(p, 0), np.array([0.0, 1.0], dtype=float)))
        self.assertTrue(np.allclose(traj.q_at(p, traj.steps - 1), np.array([2.0, 3.0], dtype=float)))

    def test_build_trajectory_map_linear(self) -> None:
        dsl = {
            "type": "linear",
            "steps": 2,
            "q_dim": 2,
            "linear": {
                "A": [
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.5,
                    0.0,
                    0.0,
                    0.5,
                ],
                "b": [0.0, 0.0, 1.0, -1.0],
            },
        }
        traj = build_trajectory_map(dsl)
        p = np.array([2.0, 4.0], dtype=float)

        self.assertEqual(traj.steps, 2)
        self.assertEqual(traj.q_dim, 2)
        self.assertEqual(traj.p_dim, 2)
        self.assertTrue(np.allclose(traj.q_at(p, 0), np.array([2.0, 4.0], dtype=float)))
        self.assertTrue(np.allclose(traj.q_at(p, 1), np.array([2.0, 1.0], dtype=float)))

    def test_state_cache_unions_required(self) -> None:
        q_var = Variable(name="q", x=np.array([0.0], dtype=float))
        pack = VariablePack([q_var])
        time = TimeGrid.single_time()

        owner = OwnerKey(owner_type="demo", owner_name="thing")
        key_a = StateKey(k=0, owner=owner, dtype="vec", field="a")
        key_b = StateKey(k=0, owner=owner, dtype="vec", field="b")

        calls: list[set[StateKey] | None] = []

        def build_state(_x_all: np.ndarray, *, required=None, **_kwargs) -> dict[StateKey, object]:
            calls.append(set(required) if required is not None else None)
            out: dict[StateKey, object] = {}
            if required is None or key_a in required:
                out[key_a] = np.array([1.0], dtype=float)
            if required is None or key_b in required:
                out[key_b] = np.array([2.0], dtype=float)
            return out

        cache = StateCache(build_state=build_state)

        cache.update_if_needed(pack, time=time, required=[key_a])
        cache.update_if_needed(pack, time=time, required=[key_a, key_b])
        cache.update_if_needed(pack, time=time, required=[key_b])

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], {key_a})
        self.assertEqual(calls[1], {key_b})

        calls.clear()
        cache.update_if_needed(pack, time=time, required=None)
        cache.update_if_needed(pack, time=time, required=[key_a])
        self.assertEqual(calls, [None])

    def test_dispatch_state_builder_registers_value_and_jac_in_one_call(self) -> None:
        class DummyRotBuilder(BackendDispatchStateBuilder):
            def __init__(self):
                super().__init__(model=None, data=None, q_var="q")
                self.resolve_calls = 0
                self.register_value_and_jac(
                    dtype=DTYPE_KINEMATICS,
                    owner_type="joint",
                    field="rot",
                    value_handler=self._handle_rot,
                    jac_handler=self._handle_rot_jac,
                )
                self.last_q = np.zeros((0,), dtype=float)

            def _update_kinematics(self, q: np.ndarray) -> None:
                self.last_q = np.asarray(q, dtype=float).reshape(-1)

            def _resolve_state_ref(self, key: StateKey):
                self.resolve_calls += 1
                return str(key.owner.owner_name)

            def _handle_rot(self, q: np.ndarray, key: StateKey, state_ref):
                del q, key
                return np.array([float(len(state_ref)), 1.0], dtype=float)

            def _handle_rot_jac(self, q: np.ndarray, key: StateKey, state_ref):
                del key, state_ref
                n = int(np.asarray(q, dtype=float).size)
                return np.eye(2, n, dtype=float)

        builder = DummyRotBuilder()
        owner = OwnerKey(owner_type="joint", owner_name="j1")
        key_rot = StateKey(k=0, owner=owner, dtype=DTYPE_KINEMATICS, field="rot", frame="world")
        key_rot_j = StateKey(k=0, owner=owner, dtype=DTYPE_KINEMATICS, field="rot_J_q", frame="world")

        ignored_owner = OwnerKey(owner_type="link", owner_name="j1")
        key_ignored = StateKey(k=0, owner=ignored_owner, dtype=DTYPE_KINEMATICS, field="rot", frame="world")

        out = builder.build_state(
            np.array([0.2, -0.1], dtype=float),
            required=[key_rot, key_rot_j, key_ignored],
        )

        self.assertIn(key_rot, out)
        self.assertIn(key_rot_j, out)
        self.assertNotIn(key_ignored, out)
        self.assertTrue(np.allclose(out[key_rot], np.array([2.0, 1.0], dtype=float)))
        self.assertTrue(np.allclose(out[key_rot_j], np.eye(2, 2, dtype=float)))
        self.assertTrue(np.allclose(builder.last_q, np.array([0.2, -0.1], dtype=float)))
        self.assertEqual(builder.resolve_calls, 1)

    def test_dispatch_state_builder_routes_handlers_by_key(self) -> None:
        class DummyDispatchBuilder(BackendDispatchStateBuilder):
            def __init__(self):
                super().__init__(model=None, data=None, q_var="q")
                self.resolve_calls = 0
                self.last_q = np.zeros((0,), dtype=float)
                self.register_handlers(
                    dtype=DTYPE_KINEMATICS,
                    owner_type="link",
                    handlers={
                        "pos": self._handle_pos,
                        "rot": self._handle_rot,
                    },
                )

            def _update_kinematics(self, q: np.ndarray) -> None:
                self.last_q = np.asarray(q, dtype=float).reshape(-1)

            def _resolve_state_ref(self, key: StateKey):
                self.resolve_calls += 1
                return str(key.owner.owner_name)

            def _handle_pos(self, q: np.ndarray, key: StateKey, state_ref):
                del q, key
                return np.array([float(len(state_ref)), 1.0], dtype=float)

            def _handle_rot(self, q: np.ndarray, key: StateKey, state_ref):
                del q, key
                return np.array([float(len(state_ref)), 2.0], dtype=float)

        builder = DummyDispatchBuilder()
        owner = OwnerKey(owner_type="link", owner_name="ee")
        key_pos = StateKey(k=0, owner=owner, dtype=DTYPE_KINEMATICS, field="pos", frame="world")
        key_rot = StateKey(k=0, owner=owner, dtype=DTYPE_KINEMATICS, field="rot", frame="world")
        key_ignored = StateKey(k=0, owner=owner, dtype=DTYPE_KINEMATICS, field="acc", frame="world")
        key_ignored_k = StateKey(k=1, owner=owner, dtype=DTYPE_KINEMATICS, field="pos", frame="world")

        out = builder.build_state(
            np.array([0.2, -0.1], dtype=float),
            required=[key_pos, key_rot, key_ignored, key_ignored_k],
        )

        self.assertIn(key_pos, out)
        self.assertIn(key_rot, out)
        self.assertNotIn(key_ignored, out)
        self.assertNotIn(key_ignored_k, out)
        self.assertTrue(np.allclose(out[key_pos], np.array([2.0, 1.0], dtype=float)))
        self.assertTrue(np.allclose(out[key_rot], np.array([2.0, 2.0], dtype=float)))
        self.assertTrue(np.allclose(builder.last_q, np.array([0.2, -0.1], dtype=float)))
        self.assertEqual(builder.resolve_calls, 2)


if __name__ == "__main__":
    unittest.main()
