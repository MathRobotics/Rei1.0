from __future__ import annotations

import unittest

import numpy as np

from eiopt.core.state_cache import OwnerKey, StateCache, StateKey
from eiopt.core.state_schema import DTYPE_KINEMATICS, jac_field
from eiopt.core.time_grid import TimeGrid
from eiopt import compile_problem, format_solve_report
from eiopt.backends._template import BackendDispatchStateBuilder
from eiopt.expr.nodes import GetStateExpr, GetVarExpr
from eiopt.solvers import solve_gauss_newton
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
