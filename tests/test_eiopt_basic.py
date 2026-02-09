from __future__ import annotations

import unittest

import numpy as np

from eiopt.core.state_cache import OwnerKey, StateCache, StateKey
from eiopt.core.state_schema import jac_field, joint_q_jac_key, joint_q_key
from eiopt.core.time_grid import TimeGrid
from eiopt import compile_problem
from eiopt.adapters import with_standard_joint_q
from eiopt.expr.nodes import GetStateExpr
from eiopt.solvers import solve_gauss_newton
from eiopt.model import Problem, DirectVectorExpr, EvalContext, L2Cost, Variable, VariablePack


class TestEiOptBasic(unittest.TestCase):
    def test_gauss_newton_solves_linear_scalar(self) -> None:
        x_var = Variable(name="x", x=np.array([0.0], dtype=float))
        pack = VariablePack([x_var])

        def value(ctx: EvalContext) -> np.ndarray:
            x = float(ctx.pack.vars[0].x[0])
            return np.array([x - 3.0], dtype=float)

        def blocks(ctx: EvalContext):
            return [np.array([[1.0]], dtype=float)]

        expr = DirectVectorExpr(name="x_minus_3", vars=[x_var], fn_value=value, fn_blocks=blocks)
        problem = Problem(variables=pack, terms=[(expr, L2Cost())])
        ctx = EvalContext(pack=pack)

        solve_gauss_newton(problem, pack, max_iters=5, ctx=ctx, tol_r=1e-14, tol_dx=1e-14)
        self.assertAlmostEqual(float(x_var.x[0]), 3.0, places=10)

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

        expr = GetStateExpr(name="get_y", vars=[q_var], key_value=key_val, key_jac_q=key_jac)
        ctx = EvalContext(pack=pack, state=cache, time=time)

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
                            "dtype": "frame",
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

        _problem, _ctx, required = compile_problem(dsl, build_state=build_state)
        fields = {k.field for k in required}
        self.assertIn("pos", fields)
        self.assertIn(jac_field("pos", var="q"), fields)
        frames = {k.frame for k in required}
        self.assertEqual(frames, {"world"})

    def test_backend_wrapper_injects_joint_q(self) -> None:
        q_var = Variable(name="q", x=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float))
        pack = VariablePack([q_var])
        time = TimeGrid(N=2, dt=0.1)  # k=0,1,2

        key_q = joint_q_key(k=1)
        key_J = joint_q_jac_key(k=1, var="q")

        wrapped = with_standard_joint_q(lambda x_all, *, pack=None, time=None, required=None: {})
        cache = StateCache(build_state=wrapped)
        cache.update_if_needed(pack, time=time, required=[key_q, key_J])

        q1 = np.asarray(cache.get(key_q), dtype=float).reshape(-1)
        J1 = np.asarray(cache.get(key_J), dtype=float)

        self.assertTrue(np.allclose(q1, np.array([3.0, 4.0], dtype=float)))

        expected_J = np.zeros((2, 6), dtype=float)
        expected_J[:, 2:4] = np.eye(2, dtype=float)
        self.assertTrue(np.allclose(J1, expected_J))

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


if __name__ == "__main__":
    unittest.main()
