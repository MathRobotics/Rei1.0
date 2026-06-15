from __future__ import annotations

import pytest

import copy
import importlib
import sys
import types

import numpy as np

from rei.core.state_schema import DTYPE_DYNAMICS, DTYPE_COORD, DTYPE_KINEMATICS, make_jac_key, make_key
from rei.core.trajectory import TrajectoryMap
from rei.optimize.reductions import build_nullspace_equality_reduction
from rei.optimize.solvers import solve

def _ensure_robokots_state_stub() -> None:
    robokots_mod = types.ModuleType("robokots")
    core_mod = types.ModuleType("robokots.core")
    state_mod = types.ModuleType("robokots.core.state")

    class StateType:  # noqa: D401
        """Minimal stub for RoboKots StateType."""

        def __init__(self, owner_type: str, owner_name: str, field: str, frame: str) -> None:
            self.owner_type = owner_type
            self.owner_name = owner_name
            self.field = field
            self.frame = frame

    state_mod.StateType = StateType
    core_mod.state = state_mod
    robokots_mod.core = core_mod

    sys.modules["robokots"] = robokots_mod
    sys.modules["robokots.core"] = core_mod
    sys.modules["robokots.core.state"] = state_mod

_ensure_robokots_state_stub()
_kots_state_mod = importlib.import_module("rei.backends.state.robotics.kots")
_kots_opt_mod = importlib.import_module("rei.optimize_backends.kots")
_traj_ioc_mod = importlib.import_module("rei.optimize_backends.trajectory_ioc")
KotsTrajectoryStateBuilder = _kots_state_mod.KotsTrajectoryStateBuilder
compile_kots_trajectory_problem = _kots_opt_mod.compile_kots_trajectory_problem
compile_trajectory_ioc_problem = _traj_ioc_mod.compile_trajectory_ioc_problem
estimate_ioc_weights = _traj_ioc_mod.estimate_ioc_weights

class _FakeKotsModel:
    def __init__(self) -> None:
        self._motion = np.zeros((6,), dtype=float)
        self.kinematics_calls = 0
        self.dynamics_calls = 0
        self.kinematics_backends = []
        self.dynamics_backends = []

    def dof(self) -> int:
        return 2

    def order(self) -> int:
        return 3

    def import_motions(self, motion) -> None:
        self._motion = np.asarray(motion, dtype=float).reshape(-1).copy()

    def kinematics(self, backend=None) -> None:
        self.kinematics_calls += 1
        self.kinematics_backends.append(backend)

    def dynamics(self, backend=None) -> None:
        self.dynamics_calls += 1
        self.dynamics_backends.append(backend)

    def _split(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.array([self._motion[0], self._motion[3]], dtype=float)
        dq = np.array([self._motion[1], self._motion[4]], dtype=float)
        ddq = np.array([self._motion[2], self._motion[5]], dtype=float)
        return q, dq, ddq

    @staticmethod
    def _state_field_name(state_ref) -> str | None:
        for attr in ("field", "field_", "data_type", "dtype"):
            value = getattr(state_ref, attr, None)
            if isinstance(value, str) and value != "":
                return value
        return None

    def state_info(self, state_ref):
        if isinstance(state_ref, tuple):
            if state_ref[0] == "total_joint" and state_ref[2] == "q":
                q, _dq, _ddq = self._split()
                return q
            raise ValueError(f"Unexpected tuple state_ref: {state_ref!r}")

        field = self._state_field_name(state_ref)
        q, dq, ddq = self._split()

        if field == "torque":
            return q + 2.0 * dq + 3.0 * ddq
        if field == "torque_d1":
            return dq + 4.0 * ddq
        if field == "torque_d2":
            return ddq
        raise ValueError(f"Unsupported field: {field!r}")

    def jacobian(self, state_ref):
        field = self._state_field_name(state_ref)

        if field == "torque":
            return np.array(
                [
                    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
                ],
                dtype=float,
            )
        if field == "torque_d1":
            return np.array(
                [
                    [0.0, 1.0, 4.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 4.0],
                ],
                dtype=float,
            )
        if field == "torque_d2":
            return np.array(
                [
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        raise ValueError(f"Unsupported jacobian field: {field!r}")


class _FakeKotsModelMatvec(_FakeKotsModel):
    def __init__(self) -> None:
        super().__init__()
        self.jacobian_calls = 0
        self.matvec_calls = 0

    def jacobian(self, state_ref):
        self.jacobian_calls += 1
        raise AssertionError("jacobian() should not be used when matvec is available")

    def _jacobian_matrix(self, state_ref):
        return super().jacobian(state_ref)

    def matvec(self, state_ref, vec):
        self.matvec_calls += 1
        J = self._jacobian_matrix(state_ref)
        v = np.asarray(vec, dtype=float).reshape(-1)
        if J.shape[1] != v.size:
            raise ValueError(f"matvec size mismatch: J={J.shape}, vec={v.size}")
        return J @ v


class _FakeKotsModelJacobianMul(_FakeKotsModel):
    def __init__(self) -> None:
        super().__init__()
        self.jacobian_calls = 0
        self.jacobian_mul_calls = 0

    def jacobian(self, state_ref):
        self.jacobian_calls += 1
        raise AssertionError("jacobian() should not be used when jacobian_mul is available")

    def _jacobian_matrix(self, state_ref):
        return super().jacobian(state_ref)

    def jacobian_mul(self, state_ref, cols):
        self.jacobian_mul_calls += 1
        J = self._jacobian_matrix(state_ref)
        C = np.asarray(cols, dtype=float)
        if C.ndim == 1:
            if J.shape[1] != C.size:
                raise ValueError(f"jacobian_mul size mismatch: J={J.shape}, vec={C.size}")
            return J @ C
        if C.ndim != 2:
            raise ValueError(f"jacobian_mul expects 1D or 2D cols, got {C.shape}")
        if J.shape[1] != C.shape[0]:
            raise ValueError(f"jacobian_mul size mismatch: J={J.shape}, cols={C.shape}")
        return J @ C


class _FakeKotsModelJacobianMulAvailable(_FakeKotsModel):
    def __init__(self) -> None:
        super().__init__()
        self.jacobian_calls = 0
        self.jacobian_mul_calls = 0

    def jacobian(self, state_ref):
        self.jacobian_calls += 1
        return super().jacobian(state_ref)

    def jacobian_mul(self, state_ref, cols):
        self.jacobian_mul_calls += 1
        J = super().jacobian(state_ref)
        return J @ np.asarray(cols, dtype=float)


class _FakeKotsModelJacobianTransposeMul(_FakeKotsModel):
    def __init__(self) -> None:
        super().__init__()
        self.jacobian_transpose_mul_calls = 0

    def jacobian_transpose_mul(self, state_ref, rhs):
        self.jacobian_transpose_mul_calls += 1
        J = super().jacobian(state_ref)
        R = np.asarray(rhs, dtype=float)
        return J.T @ R


class _FakeKotsModelJacobianTransposeMulNoDenseJac(_FakeKotsModel):
    def __init__(self) -> None:
        super().__init__()
        self.jacobian_calls = 0
        self.jacobian_transpose_mul_calls = 0

    def jacobian(self, state_ref):
        self.jacobian_calls += 1
        raise AssertionError("jacobian() should not be used by stationarity VJP fast path")

    def _jacobian_matrix(self, state_ref):
        return _FakeKotsModel.jacobian(self, state_ref)

    def jacobian_transpose_mul(self, state_ref, rhs):
        self.jacobian_transpose_mul_calls += 1
        J = self._jacobian_matrix(state_ref)
        R = np.asarray(rhs, dtype=float)
        return J.T @ R


class _FakeKotsModelOrder4:
    def __init__(self) -> None:
        self._motion = np.zeros((8,), dtype=float)

    def dof(self) -> int:
        return 2

    def order(self) -> int:
        return 4

    def import_motions(self, motion) -> None:
        self._motion = np.asarray(motion, dtype=float).reshape(-1).copy()

    def kinematics(self) -> None:
        return None

    def dynamics(self) -> None:
        return None

    @staticmethod
    def _state_field_name(state_ref) -> str | None:
        for attr in ("field", "field_", "data_type", "dtype"):
            value = getattr(state_ref, attr, None)
            if isinstance(value, str) and value != "":
                return value
        return None

    def _split(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = np.array([self._motion[0], self._motion[4]], dtype=float)
        dq = np.array([self._motion[1], self._motion[5]], dtype=float)
        ddq = np.array([self._motion[2], self._motion[6]], dtype=float)
        d3q = np.array([self._motion[3], self._motion[7]], dtype=float)
        return q, dq, ddq, d3q

    def state_info(self, state_ref):
        field = self._state_field_name(state_ref)
        q, dq, ddq, _d3q = self._split()
        if field == "torque":
            return q + 2.0 * dq + 3.0 * ddq
        raise ValueError(f"Unsupported field: {field!r}")

    def jacobian(self, state_ref):
        field = self._state_field_name(state_ref)
        if field == "torque":
            # Intentionally uses only q,dq,ddq columns (6), despite model order=4.
            return np.array(
                [
                    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
                ],
                dtype=float,
            )
        raise ValueError(f"Unsupported jacobian field: {field!r}")

class _FakeKotsModelLinkLocalJacobian:
    def __init__(self) -> None:
        self._motion = np.zeros((6,), dtype=float)

    def dof(self) -> int:
        return 2

    def order(self) -> int:
        return 3

    def import_motions(self, motion) -> None:
        self._motion = np.asarray(motion, dtype=float).reshape(-1).copy()

    def kinematics(self) -> None:
        return None

    @staticmethod
    def _state_field_name(state_ref) -> str | None:
        for attr in ("field", "field_", "data_type", "dtype"):
            value = getattr(state_ref, attr, None)
            if isinstance(value, str) and value != "":
                return value
        return None

    def state_info(self, state_ref):
        field = self._state_field_name(state_ref)
        q0 = float(self._motion[0])
        q1 = float(self._motion[3])
        theta = q0 + q1
        c = float(np.cos(theta))
        s = float(np.sin(theta))

        if field == "rot":
            return np.array(
                [
                    [c, -s, 0.0],
                    [s, c, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            ).reshape(-1)
        if field == "pos":
            return np.array([0.0, 0.0, 0.0], dtype=float)
        raise ValueError(f"Unsupported field: {field!r}")

    def jacobian(self, state_ref):
        field = self._state_field_name(state_ref)
        if field != "pos":
            raise ValueError(f"Unsupported jacobian field: {field!r}")

        # Local-frame Jacobian for q=[0.3, -0.4] style sample.
        return np.array(
            [
                [-0.3894183423086506, 0.0],
                [1.921060994002885, 1.0],
                [0.0, 0.0],
            ],
            dtype=float,
        )

def _traj_map_from_rows(rows: list[list[float]]) -> TrajectoryMap:
    A = np.asarray(rows, dtype=float)
    b = np.zeros((A.shape[0],), dtype=float)
    return TrajectoryMap(A=A, b=b, steps=2, q_dim=2)

class TestKotsTrajectoryDynamicsMock:
    def test_kots_link_pos_jacobian_is_rotated_to_world_frame(self) -> None:
        model = _FakeKotsModelLinkLocalJacobian()
        builder = _kots_state_mod.KotsStateBuilder(
            model,
            data={},
            q_var="q",
            fields=("pos",),
            dynamics_fields=None,
        )

        key = make_jac_key(
            k=0,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
            var="q",
        )
        q = np.array([0.3, -0.4], dtype=float)
        state = builder.build_state(q, required=[key])
        J_world = np.asarray(state[key], dtype=float)

        theta = float(np.sum(q))
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        rot_world = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        J_local = np.array(
            [
                [-0.3894183423086506, 0.0],
                [1.921060994002885, 1.0],
                [0.0, 0.0],
            ],
            dtype=float,
        )
        expected = rot_world @ J_local
        np.testing.assert_allclose(J_world, expected, rtol=0.0, atol=1e-12)

    def test_kots_state_field_name_keeps_canonical_torque_derivative_orders(self) -> None:
        assert _kots_state_mod.KotsAdapter.state_field_name("torque_d1") == "torque_d1"
        assert _kots_state_mod.KotsAdapter.state_field_name("torque_d2") == "torque_d2"
        with pytest.raises(ValueError, match="unsupported field alias"):
            _ = _kots_state_mod.KotsAdapter.state_field_name("tau_diff2")

    def test_kots_state_ref_backend_fallback_uses_torque_diff_for_derivative(self) -> None:
        original_state_type = _kots_state_mod.StateType

        class _StrictStateType:
            def __init__(self, owner_type: str, owner_name: str, field: str, frame: str | None) -> None:
                if field == "torque_d1":
                    raise KeyError(field)
                self.owner_type = owner_type
                self.owner_name = owner_name
                self.field = field
                self.frame = frame

        _kots_state_mod.StateType = _StrictStateType
        try:
            builder = _kots_state_mod.KotsStateBuilder(
                _FakeKotsModel(),
                data={},
                dynamics_fields=("torque_d1",),
            )
            key = make_key(
                k=0,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="torque_d1",
            )
            state_ref = builder._resolve_state_ref(key)
            assert getattr(state_ref, "field", None) == "torque_diff1"
        finally:
            _kots_state_mod.StateType = original_state_type

    def test_compile_kots_trajectory_problem_builds_runtime_bundle(self) -> None:
        model = _FakeKotsModel()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 2,
                "A": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ],
            },
            "variables": [
                {"name": "p", "dim": 2, "init": [0.0, 0.0]},
            ],
            "terms": [
                {
                    "expr": {"type": "get_var", "name": "p_identity", "var": "p"},
                    "cost": {"type": "l2"},
                }
            ],
        }

        compiled = compile_kots_trajectory_problem(
            dsl,
            model=model,
            data={},
            dynamics_fields=("torque",),
        )

        assert compiled.p_var == "p"
        assert compiled.dt == pytest.approx(0.2, rel=0.0, abs=1e-7)
        assert compiled.model_order == 3
        assert compiled.trajectory_map.steps == 2
        assert compiled.trajectory_map.q_dim == 2
        assert compiled.trajectory_map.p_dim == 2
        assert sorted(compiled.trajectory_derivative_maps.keys()) == [0, 1, 2]
        assert compiled.runtime.pack.n_total == 2

        r, J = compiled.runtime.linearize()
        assert np.allclose(r, np.array([0.0, 0.0], dtype=float))
        assert np.allclose(J, np.eye(2, dtype=float))

    def test_compile_trajectory_ioc_problem_kots_uses_matvec_for_param_jacobian(self) -> None:
        model = _FakeKotsModelMatvec()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 2,
                "A": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ],
            },
            "variables": [
                {"name": "p", "dim": 2, "init": [0.5, -0.25]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "name": "torque0",
                        "key": {
                            "k": 0,
                            "owner_type": "total_joint",
                            "owner_name": "robot",
                            "dtype": DTYPE_DYNAMICS,
                            "field": "torque",
                        },
                        "jac": {"var": "p"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        with pytest.warns(DeprecationWarning, match="prefer_matvec_jacobian"):
            compiled = compile_trajectory_ioc_problem(
                dsl,
                backend="kots",
                model=model,
                data={},
                prefer_matvec_jacobian=True,
            )
        result = estimate_ioc_weights(compiled)

        assert result["backend"] == "kots"
        assert model.matvec_calls > 0
        assert model.jacobian_calls == 0

    def test_compile_trajectory_ioc_problem_kots_uses_jacobian_mul_for_param_jacobian(self) -> None:
        model = _FakeKotsModelJacobianMul()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 2,
                "A": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ],
            },
            "variables": [
                {"name": "p", "dim": 2, "init": [0.5, -0.25]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "name": "torque0",
                        "key": {
                            "k": 0,
                            "owner_type": "total_joint",
                            "owner_name": "robot",
                            "dtype": DTYPE_DYNAMICS,
                            "field": "torque",
                        },
                        "jac": {"var": "p"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        with pytest.warns(DeprecationWarning, match="prefer_matvec_jacobian"):
            compiled = compile_trajectory_ioc_problem(
                dsl,
                backend="kots",
                model=model,
                data={},
                prefer_matvec_jacobian=True,
            )
        result = estimate_ioc_weights(compiled)

        assert result["backend"] == "kots"
        assert model.jacobian_mul_calls > 0
        assert model.jacobian_calls == 0

    def test_kots_trajectory_default_jacobian_strategy_uses_jacobian_mul(self) -> None:
        model = _FakeKotsModelJacobianMul()
        trajectory_map = TrajectoryMap.from_blocks(
            [
                np.eye(2, dtype=float),
                2.0 * np.eye(2, dtype=float),
            ]
        )
        builder = KotsTrajectoryStateBuilder(
            model,
            {},
            trajectory_map=trajectory_map,
        )
        jac_key = make_jac_key(
            k=0,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_DYNAMICS,
            field="torque",
            var="p",
        )

        J = builder.build_state(np.array([0.5, -0.25], dtype=float), required=[jac_key])[jac_key]

        assert np.allclose(J, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float))
        assert model.jacobian_mul_calls > 0
        assert model.jacobian_calls == 0

    def test_kots_trajectory_jacobian_strategy_dense_uses_dense_jacobian(self) -> None:
        model = _FakeKotsModelJacobianMulAvailable()
        trajectory_map = TrajectoryMap.from_blocks(
            [
                np.eye(2, dtype=float),
                2.0 * np.eye(2, dtype=float),
            ]
        )
        builder = KotsTrajectoryStateBuilder(
            model,
            {},
            trajectory_map=trajectory_map,
            jacobian_strategy="dense",
        )
        jac_key = make_jac_key(
            k=0,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_DYNAMICS,
            field="torque",
            var="p",
        )

        J = builder.build_state(np.array([0.5, -0.25], dtype=float), required=[jac_key])[jac_key]

        assert np.allclose(J, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float))
        assert model.jacobian_calls > 0
        assert model.jacobian_mul_calls == 0

    def test_kots_trajectory_jacobian_strategy_rejects_auto(self) -> None:
        model = _FakeKotsModelJacobianMulAvailable()
        trajectory_map = TrajectoryMap.from_blocks(
            [
                np.eye(2, dtype=float),
                2.0 * np.eye(2, dtype=float),
            ]
        )

        with pytest.raises(ValueError, match="jacobian_strategy"):
            KotsTrajectoryStateBuilder(
                model,
                {},
                trajectory_map=trajectory_map,
                jacobian_strategy="auto",
            )

    def test_kots_trajectory_builder_passes_kots_backend_to_dynamics_update(self) -> None:
        model = _FakeKotsModel()
        trajectory_map = TrajectoryMap.from_blocks(
            [
                np.eye(2, dtype=float),
                2.0 * np.eye(2, dtype=float),
            ]
        )
        builder = KotsTrajectoryStateBuilder(
            model,
            {},
            trajectory_map=trajectory_map,
            kots_backend="rust",
        )
        key = make_key(
            k=0,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_DYNAMICS,
            field="torque",
        )

        out = builder.build_state(np.array([0.5, -0.25], dtype=float), required=[key])

        assert np.allclose(out[key], np.array([0.5, -0.25], dtype=float))
        assert model.dynamics_backends
        assert set(model.dynamics_backends) == {"rust"}

    def test_compile_kots_trajectory_problem_passes_kots_backend_to_builder(self) -> None:
        model = _FakeKotsModel()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 2,
                "A": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ],
            },
            "variables": [
                {"name": "p", "dim": 2, "init": [0.5, -0.25]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "name": "torque0",
                        "key": {
                            "k": 0,
                            "owner_type": "total_joint",
                            "owner_name": "robot",
                            "dtype": DTYPE_DYNAMICS,
                            "field": "torque",
                        },
                        "jac": {"var": "p"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        compiled = compile_kots_trajectory_problem(
            dsl,
            model=model,
            data={},
            kots_backend="rust",
        )

        compiled.runtime.linearize()

        assert model.dynamics_backends
        assert set(model.dynamics_backends) == {"rust"}

    def test_kots_trajectory_builder_rejects_unknown_kots_backend(self) -> None:
        model = _FakeKotsModel()
        trajectory_map = TrajectoryMap.from_blocks([np.eye(2, dtype=float)])

        with pytest.raises(ValueError, match="kots_backend"):
            KotsTrajectoryStateBuilder(
                model,
                {},
                trajectory_map=trajectory_map,
                kots_backend="cuda",
            )

    def test_kots_trajectory_param_jacobian_transpose_mul_uses_backend_api(self) -> None:
        model = _FakeKotsModelJacobianTransposeMul()
        trajectory_map = TrajectoryMap.from_blocks(
            [
                np.array(
                    [
                        [1.0, 0.5],
                        [-0.25, 2.0],
                    ],
                    dtype=float,
                ),
                np.array(
                    [
                        [0.0, 1.0],
                        [1.0, 0.0],
                    ],
                    dtype=float,
                ),
            ]
        )
        builder = KotsTrajectoryStateBuilder(
            model,
            {},
            trajectory_map=trajectory_map,
        )
        key = make_key(
            k=0,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_DYNAMICS,
            field="torque",
        )
        jac_key = make_jac_key(
            k=0,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_DYNAMICS,
            field="torque",
            var="p",
        )
        p = np.array([0.25, -0.5], dtype=float)
        rhs = np.array([4.0, -2.0], dtype=float)

        out = builder.param_jacobian_transpose_mul(p, key, rhs)
        Jp = np.asarray(builder.build_state(p, required=[jac_key])[jac_key], dtype=float)

        assert np.allclose(out, Jp.T @ rhs)
        assert model.jacobian_transpose_mul_calls > 0

    def test_compile_trajectory_ioc_problem_kots_uses_transpose_mul_for_stationarity(self) -> None:
        model = _FakeKotsModelJacobianTransposeMulNoDenseJac()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 2,
                "A": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ],
            },
            "variables": [
                {"name": "p", "dim": 2, "init": [0.5, -0.25]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "name": "torque0",
                        "key": {
                            "k": 0,
                            "owner_type": "total_joint",
                            "owner_name": "robot",
                            "dtype": DTYPE_DYNAMICS,
                            "field": "torque",
                        },
                        "jac": {"var": "p"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        compiled = compile_trajectory_ioc_problem(
            dsl,
            backend="kots",
            model=model,
            data={},
        )
        result = estimate_ioc_weights(compiled)

        assert result["backend"] == "kots"
        assert model.jacobian_transpose_mul_calls > 0
        assert model.jacobian_calls == 0

    def test_compile_kots_trajectory_problem_supports_external_nullspace_eq_runtime(self) -> None:
        model = _FakeKotsModel()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 2,
                "A": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "variables": [
                {"name": "p", "dim": 4, "init": [0.0, 0.0, 0.0, 0.0]},
            ],
            "terms": [
                {
                    "constraint": {"kind": "eq"},
                    "expr": {
                        "type": "sub",
                        "name": "q_init_eq",
                        "a": {
                            "type": "get_state",
                            "key": {
                                "k": 0,
                                "owner_type": "total_joint",
                                "owner_name": "robot",
                                "dtype": DTYPE_COORD,
                                "field": "q",
                            },
                            "jac": {"var": "p"},
                        },
                        "b": {
                            "type": "const",
                            "var": "p",
                            "value": [1.0, 2.0],
                        },
                    },
                    "cost": {"type": "l2"},
                },
                {
                    "expr": {"type": "get_var", "name": "p_identity", "var": "p"},
                    "cost": {"type": "l2"},
                },
            ],
        }

        compiled = compile_kots_trajectory_problem(
            dsl,
            model=model,
            data={},
        )
        reduction = build_nullspace_equality_reduction(compiled.runtime)
        assert reduction is not None
        assert reduction is not None
        assert reduction.runtime.pack.n_total == 2
        assert compiled.runtime.pack.n_total == 4
        runtime_for_solve = reduction.runtime
        assert runtime_for_solve is reduction.runtime

        out = solve(
            runtime_for_solve,
            solver="gauss_newton",
            options={"max_iters": 30, "tol_r": 1e-12, "tol_dx": 1e-12},
        )
        z_star = out.solution
        assert out.converged
        p_star = reduction.lift(z_star)
        assert np.allclose(p_star, np.array([1.0, 2.0, 0.0, 0.0], dtype=float), atol=1e-8)

        p_cur = compiled.runtime.pack.get().copy()
        assert np.allclose(p_cur, p_star, atol=1e-12)
        eq_terms = compiled.runtime.linearize_constraint_terms(kind="eq", weighted=False)
        req = np.concatenate([term.residual for term in eq_terms], axis=0)
        assert float(np.linalg.norm(req)) < 1e-10

    def test_compile_kots_trajectory_problem_nullspace_term_selection(self) -> None:
        model = _FakeKotsModel()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 2,
                "A": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "variables": [
                {"name": "p", "dim": 4, "init": [0.0, 0.0, 0.0, 0.0]},
            ],
            "terms": [
                {
                    "constraint": {"kind": "eq"},
                    "expr": {
                        "type": "sub",
                        "name": "q0_eq",
                        "a": {
                            "type": "get_state",
                            "key": {
                                "k": 0,
                                "owner_type": "total_joint",
                                "owner_name": "robot",
                                "dtype": DTYPE_COORD,
                                "field": "q",
                            },
                            "jac": {"var": "p"},
                        },
                        "b": {
                            "type": "const",
                            "var": "p",
                            "value": [1.0, 2.0],
                        },
                    },
                    "cost": {"type": "l2"},
                },
                {
                    "constraint": {"kind": "eq"},
                    "expr": {
                        "type": "sub",
                        "name": "q1_eq",
                        "a": {
                            "type": "get_state",
                            "key": {
                                "k": 1,
                                "owner_type": "total_joint",
                                "owner_name": "robot",
                                "dtype": DTYPE_COORD,
                                "field": "q",
                            },
                            "jac": {"var": "p"},
                        },
                        "b": {
                            "type": "const",
                            "var": "p",
                            "value": [3.0, 4.0],
                        },
                    },
                    "cost": {"type": "l2"},
                },
                {
                    "expr": {"type": "get_var", "name": "p_identity", "var": "p"},
                    "cost": {"type": "l2"},
                },
            ],
        }

        compiled = compile_kots_trajectory_problem(
            dsl,
            model=model,
            data={},
        )
        reduction = build_nullspace_equality_reduction(
            compiled.runtime,
            eq_term_indices=[0],
            objective_term_indices=[2],
        )
        assert reduction is not None
        assert reduction is not None
        assert reduction.eq_term_indices == (0,)
        assert reduction.objective_term_indices == (2,)
        assert reduction.runtime.pack.n_total == 2

        selected = reduction.runtime.linearize_terms(weighted=False, term_indices=[2])
        assert [t.term_index for t in selected] == [2]
        with pytest.raises(ValueError, match="global problem indexing"):
            reduction.runtime.linearize_terms(weighted=False, term_indices=[1])

    def test_compile_kots_trajectory_problem_nullspace_rejects_non_eq_term_selection(self) -> None:
        model = _FakeKotsModel()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 2,
                "A": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "variables": [
                {"name": "p", "dim": 4, "init": [0.0, 0.0, 0.0, 0.0]},
            ],
            "terms": [
                {
                    "constraint": {"kind": "eq"},
                    "expr": {
                        "type": "sub",
                        "name": "q_init_eq",
                        "a": {
                            "type": "get_state",
                            "key": {
                                "k": 0,
                                "owner_type": "total_joint",
                                "owner_name": "robot",
                                "dtype": DTYPE_COORD,
                                "field": "q",
                            },
                            "jac": {"var": "p"},
                        },
                        "b": {
                            "type": "const",
                            "var": "p",
                            "value": [1.0, 2.0],
                        },
                    },
                    "cost": {"type": "l2"},
                },
                {
                    "expr": {"type": "get_var", "name": "p_identity", "var": "p"},
                    "cost": {"type": "l2"},
                },
            ],
        }

        with pytest.raises(ValueError, match="constraint.kind='eq'"):
            _ = build_nullspace_equality_reduction(
                compile_kots_trajectory_problem(
                    dsl,
                    model=model,
                    data={},
                ).runtime,
                eq_term_indices=[1],
            )

    def test_compile_kots_trajectory_problem_rejects_legacy_enabled_argument(self) -> None:
        model = _FakeKotsModel()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 2,
                "A": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "variables": [
                {"name": "p", "dim": 4, "init": [0.0, 0.0, 0.0, 0.0]},
            ],
            "terms": [
                {
                    "constraint": {"kind": "eq"},
                    "expr": {
                        "type": "sub",
                        "name": "q_init_eq",
                        "a": {
                            "type": "get_state",
                            "key": {
                                "k": 0,
                                "owner_type": "total_joint",
                                "owner_name": "robot",
                                "dtype": DTYPE_COORD,
                                "field": "q",
                            },
                            "jac": {"var": "p"},
                        },
                        "b": {
                            "type": "const",
                            "var": "p",
                            "value": [1.0, 2.0],
                        },
                    },
                    "cost": {"type": "l2"},
                },
                {
                    "expr": {"type": "get_var", "name": "p_identity", "var": "p"},
                    "cost": {"type": "l2"},
                },
            ],
        }

        with pytest.raises(TypeError, match="enabled"):
            _ = build_nullspace_equality_reduction(
                compile_kots_trajectory_problem(
                    dsl,
                    model=model,
                    data={},
                ).runtime,
                enabled=False,  # type: ignore[call-arg]
                eq_term_indices=[0],
            )

    def test_compile_kots_trajectory_problem_auto_fills_p_dim_and_fill_init(self) -> None:
        model = _FakeKotsModel()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "bspline",
                "var": "p",
                "steps": 2,
                "degree": 3,
                "num_ctrl_points": 4,
            },
            "variables": [
                {"name": "p", "init": {"fill": 0.0}},
            ],
            "terms": [
                {
                    "expr": {"type": "get_var", "name": "p_identity", "var": "p"},
                    "cost": {"type": "l2"},
                }
            ],
        }

        compiled = compile_kots_trajectory_problem(
            dsl,
            model=model,
            data={},
        )
        assert compiled.trajectory_map.q_dim == 2
        assert compiled.trajectory_map.p_dim == 8
        assert compiled.runtime.pack.n_total == 8
        assert np.allclose(compiled.runtime.pack.get(), np.zeros((8,), dtype=float))

    def test_compile_kots_trajectory_problem_does_not_mutate_input_dsl(self) -> None:
        model = _FakeKotsModel()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "bspline",
                "var": "p",
                "steps": 2,
                "degree": 3,
                "num_ctrl_points": 4,
            },
            "variables": [
                {"name": "p", "init": {"fill": 0.0}},
            ],
            "terms": [
                {
                    "expr": {"type": "get_var", "name": "p_identity", "var": "p"},
                    "cost": {"type": "l2"},
                }
            ],
        }
        dsl_before = copy.deepcopy(dsl)

        _ = compile_kots_trajectory_problem(
            dsl,
            model=model,
            data={},
        )
        assert dsl == dsl_before
        assert "q_dim" not in dsl["trajectory"]
        assert "dim" not in dsl["variables"][0]

    def test_compile_kots_trajectory_problem_rejects_scalar_init_without_fill(self) -> None:
        model = _FakeKotsModel()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "bspline",
                "var": "p",
                "steps": 2,
                "degree": 3,
                "num_ctrl_points": 4,
            },
            "variables": [
                {"name": "p", "init": 0.0},
            ],
            "terms": [
                {
                    "expr": {"type": "get_var", "name": "p_identity", "var": "p"},
                    "cost": {"type": "l2"},
                }
            ],
        }

        with pytest.raises(ValueError, match="init = \\{ fill = <value> \\}"):
            _ = compile_kots_trajectory_problem(
                dsl,
                model=model,
                data={},
            )

    def test_compile_kots_trajectory_problem_inferrs_dynamics_fields_from_dsl(self) -> None:
        model = _FakeKotsModel()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 2,
                "A": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ],
            },
            "variables": [
                {"name": "p", "dim": 2, "init": [0.0, 0.0]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "key": {
                            "k": 0,
                            "owner_type": "total_joint",
                            "owner_name": "robot",
                            "dtype": DTYPE_DYNAMICS,
                            "field": "torque",
                        },
                        "jac": {"var": "p"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        compiled = compile_kots_trajectory_problem(
            dsl,
            model=model,
            data={},
        )
        assert compiled.dynamics_fields == ("torque",)
        _ = compiled.runtime.linearize()

    def test_compile_kots_trajectory_problem_validates_p_dim(self) -> None:
        model = _FakeKotsModel()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 2,
                "A": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ],
            },
            "variables": [
                {"name": "p", "dim": 3, "init": [0.0, 0.0, 0.0]},
            ],
            "terms": [
                {
                    "expr": {"type": "get_var", "name": "p_identity", "var": "p"},
                    "cost": {"type": "l2"},
                }
            ],
        }

        with pytest.raises(ValueError, match="dim mismatch"):
            _ = compile_kots_trajectory_problem(
                dsl,
                model=model,
                data={},
            )

    def test_compile_kots_trajectory_problem_detects_missing_dynamics_field_registration(self) -> None:
        model = _FakeKotsModel()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 2,
                "A": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ],
            },
            "variables": [
                {"name": "p", "dim": 2, "init": [0.0, 0.0]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "key": {
                            "k": 0,
                            "owner_type": "total_joint",
                            "owner_name": "robot",
                            "dtype": DTYPE_DYNAMICS,
                            "field": "torque_d1",
                        },
                        "jac": {"var": "p"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        with pytest.raises(ValueError, match="Missing: torque_d1"):
            _ = compile_kots_trajectory_problem(
                dsl,
                model=model,
                data={},
                dynamics_fields=("torque",),
            )

    def test_compile_kots_trajectory_problem_detects_unsupported_dynamics_owner_type(self) -> None:
        model = _FakeKotsModel()
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 2,
                "A": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ],
            },
            "variables": [
                {"name": "p", "dim": 2, "init": [0.0, 0.0]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "key": {
                            "k": 0,
                            "owner_type": "joint",
                            "owner_name": "joint0",
                            "dtype": DTYPE_DYNAMICS,
                            "field": "torque",
                        },
                        "jac": {"var": "p"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        with pytest.raises(ValueError, match="unsupported owner_type"):
            _ = compile_kots_trajectory_problem(
                dsl,
                model=model,
                data={},
                dynamics_fields=("torque",),
                dynamics_owner_type="total_joint",
            )

    def test_dynamics_value_and_param_jac_chain(self) -> None:
        model = _FakeKotsModel()
        traj0 = _traj_map_from_rows(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [6.0, 0.0],
                [0.0, 7.0],
            ]
        )
        traj1 = _traj_map_from_rows(
            [
                [2.0, 0.0],
                [0.0, 3.0],
                [8.0, 0.0],
                [0.0, 9.0],
            ]
        )
        traj2 = _traj_map_from_rows(
            [
                [4.0, 0.0],
                [0.0, 5.0],
                [10.0, 0.0],
                [0.0, 11.0],
            ]
        )

        builder = KotsTrajectoryStateBuilder(
            model,
            data={},
            trajectory_map=traj0,
            trajectory_derivative_maps={0: traj0, 1: traj1, 2: traj2},
            p_var="p",
            dynamics_fields=("torque", "torque_d1"),
        )

        required = [
            make_key(
                k=0,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="torque",
            ),
            make_jac_key(
                k=0,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="torque",
                var="p",
            ),
            make_key(
                k=0,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="torque_d1",
            ),
            make_jac_key(
                k=0,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="torque_d1",
                var="p",
            ),
            make_jac_key(
                k=0,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_COORD,
                field="q",
                var="p",
            ),
            make_key(
                k=1,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="torque",
            ),
            make_jac_key(
                k=1,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="torque",
                var="p",
            ),
            make_key(
                k=1,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="torque_d1",
            ),
            make_jac_key(
                k=1,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="torque_d1",
                var="p",
            ),
            make_jac_key(
                k=1,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_COORD,
                field="q",
                var="p",
            ),
        ]

        p = np.array([1.0, 2.0], dtype=float)
        out = builder.build_state(p, required=required)

        assert np.allclose(
            out[make_key(k=0, owner_type="total_joint", owner_name="robot", dtype=DTYPE_DYNAMICS, field="torque")],
            np.array([17.0, 44.0], dtype=float),
        )
        assert np.allclose(
            out[
                make_key(
                    k=0,
                    owner_type="total_joint",
                    owner_name="robot",
                    dtype=DTYPE_DYNAMICS,
                    field="torque_d1",
                )
            ],
            np.array([18.0, 46.0], dtype=float),
        )
        assert np.allclose(
            out[make_key(k=1, owner_type="total_joint", owner_name="robot", dtype=DTYPE_DYNAMICS, field="torque")],
            np.array([52.0, 116.0], dtype=float),
        )
        assert np.allclose(
            out[
                make_key(
                    k=1,
                    owner_type="total_joint",
                    owner_name="robot",
                    dtype=DTYPE_DYNAMICS,
                    field="torque_d1",
                )
            ],
            np.array([48.0, 106.0], dtype=float),
        )

        assert np.allclose(
            out[
                make_jac_key(
                    k=0,
                    owner_type="total_joint",
                    owner_name="robot",
                    dtype=DTYPE_DYNAMICS,
                    field="torque",
                    var="p",
                )
            ],
            np.array([[17.0, 0.0], [0.0, 22.0]], dtype=float),
        )
        assert np.allclose(
            out[
                make_jac_key(
                    k=0,
                    owner_type="total_joint",
                    owner_name="robot",
                    dtype=DTYPE_DYNAMICS,
                    field="torque_d1",
                    var="p",
                )
            ],
            np.array([[18.0, 0.0], [0.0, 23.0]], dtype=float),
        )
        assert np.allclose(
            out[
                make_jac_key(
                    k=1,
                    owner_type="total_joint",
                    owner_name="robot",
                    dtype=DTYPE_DYNAMICS,
                    field="torque",
                    var="p",
                )
            ],
            np.array([[52.0, 0.0], [0.0, 58.0]], dtype=float),
        )
        assert np.allclose(
            out[
                make_jac_key(
                    k=1,
                    owner_type="total_joint",
                    owner_name="robot",
                    dtype=DTYPE_DYNAMICS,
                    field="torque_d1",
                    var="p",
                )
            ],
            np.array([[48.0, 0.0], [0.0, 53.0]], dtype=float),
        )

        assert np.allclose(
            out[
                make_jac_key(
                    k=0,
                    owner_type="total_joint",
                    owner_name="robot",
                    dtype=DTYPE_COORD,
                    field="q",
                    var="p",
                )
            ],
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        )
        assert np.allclose(
            out[
                make_jac_key(
                    k=1,
                    owner_type="total_joint",
                    owner_name="robot",
                    dtype=DTYPE_COORD,
                    field="q",
                    var="p",
                )
            ],
            np.array([[6.0, 0.0], [0.0, 7.0]], dtype=float),
        )

        assert model.kinematics_calls == 0
        assert model.dynamics_calls == 2

    def test_second_torque_derivative_value_and_jac_chain(self) -> None:
        model = _FakeKotsModel()
        traj0 = _traj_map_from_rows(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [6.0, 0.0],
                [0.0, 7.0],
            ]
        )
        traj1 = _traj_map_from_rows(
            [
                [2.0, 0.0],
                [0.0, 3.0],
                [8.0, 0.0],
                [0.0, 9.0],
            ]
        )
        traj2 = _traj_map_from_rows(
            [
                [4.0, 0.0],
                [0.0, 5.0],
                [10.0, 0.0],
                [0.0, 11.0],
            ]
        )

        builder = KotsTrajectoryStateBuilder(
            model,
            data={},
            trajectory_map=traj0,
            trajectory_derivative_maps={0: traj0, 1: traj1, 2: traj2},
            p_var="p",
            dynamics_fields=("torque", "torque_d1", "torque_d2"),
        )

        key_d2 = make_key(
            k=0,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_DYNAMICS,
            field="torque_d2",
        )
        key_d2_jac = make_jac_key(
            k=0,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_DYNAMICS,
            field="torque_d2",
            var="p",
        )

        out = builder.build_state(np.array([1.0, 2.0], dtype=float), required=[key_d2, key_d2_jac])
        assert np.allclose(out[key_d2], np.array([4.0, 10.0], dtype=float))
        assert np.allclose(out[key_d2_jac], np.array([[4.0, 0.0], [0.0, 5.0]], dtype=float))

    def test_order4_low_order_state_jac_chain(self) -> None:
        model = _FakeKotsModelOrder4()
        traj0 = _traj_map_from_rows(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [6.0, 0.0],
                [0.0, 7.0],
            ]
        )
        traj1 = _traj_map_from_rows(
            [
                [2.0, 0.0],
                [0.0, 3.0],
                [8.0, 0.0],
                [0.0, 9.0],
            ]
        )
        traj2 = _traj_map_from_rows(
            [
                [4.0, 0.0],
                [0.0, 5.0],
                [10.0, 0.0],
                [0.0, 11.0],
            ]
        )
        traj3 = _traj_map_from_rows(
            [
                [12.0, 0.0],
                [0.0, 13.0],
                [14.0, 0.0],
                [0.0, 15.0],
            ]
        )

        builder = KotsTrajectoryStateBuilder(
            model,
            data={},
            trajectory_map=traj0,
            trajectory_derivative_maps={0: traj0, 1: traj1, 2: traj2, 3: traj3},
            p_var="p",
            dynamics_fields=("torque",),
        )

        key_tau = make_key(
            k=0,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_DYNAMICS,
            field="torque",
        )
        key_tau_jac = make_jac_key(
            k=0,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_DYNAMICS,
            field="torque",
            var="p",
        )

        out = builder.build_state(np.array([1.0, 2.0], dtype=float), required=[key_tau, key_tau_jac])
        assert np.allclose(out[key_tau], np.array([17.0, 44.0], dtype=float))
        assert np.allclose(out[key_tau_jac], np.array([[17.0, 0.0], [0.0, 22.0]], dtype=float))
