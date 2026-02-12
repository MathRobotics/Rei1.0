from __future__ import annotations

import importlib
import sys
import types
import unittest

import numpy as np

from eiopt.core.state_schema import DTYPE_DYNAMICS, DTYPE_JOINT, make_jac_key, make_key
from eiopt.core.trajectory import TrajectoryMap


def _ensure_robokots_state_stub() -> None:
    try:
        from robokots.core.state import StateType  # noqa: F401

        return
    except Exception:
        pass

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
_kots_mod = importlib.import_module("eiopt.backends.kots")
KotsTrajectoryStateBuilder = _kots_mod.KotsTrajectoryStateBuilder


class _FakeKotsModel:
    def __init__(self) -> None:
        self._motion = np.zeros((6,), dtype=float)
        self.kinematics_calls = 0
        self.dynamics_calls = 0

    def dof(self) -> int:
        return 2

    def order(self) -> int:
        return 3

    def import_motions(self, motion) -> None:
        self._motion = np.asarray(motion, dtype=float).reshape(-1).copy()

    def kinematics(self) -> None:
        self.kinematics_calls += 1

    def dynamics(self) -> None:
        self.dynamics_calls += 1

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
        if field in ("torque_rate", "torque_diff1"):
            return dq + 4.0 * ddq
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
        if field in ("torque_rate", "torque_diff1"):
            return np.array(
                [
                    [0.0, 1.0, 4.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 4.0],
                ],
                dtype=float,
            )
        raise ValueError(f"Unsupported jacobian field: {field!r}")


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


def _traj_map_from_rows(rows: list[list[float]]) -> TrajectoryMap:
    A = np.asarray(rows, dtype=float)
    b = np.zeros((A.shape[0],), dtype=float)
    return TrajectoryMap(A=A, b=b, steps=2, q_dim=2)


class TestKotsTrajectoryDynamicsMock(unittest.TestCase):
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
            dynamics_fields=("tau", "dtau"),
        )

        required = [
            make_key(
                k=0,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="tau",
            ),
            make_jac_key(
                k=0,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="tau",
                var="p",
            ),
            make_key(
                k=0,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="dtau",
            ),
            make_jac_key(
                k=0,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="dtau",
                var="p",
            ),
            make_jac_key(
                k=0,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_JOINT,
                field="q",
                var="p",
            ),
            make_key(
                k=1,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="tau",
            ),
            make_jac_key(
                k=1,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="tau",
                var="p",
            ),
            make_key(
                k=1,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="dtau",
            ),
            make_jac_key(
                k=1,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_DYNAMICS,
                field="dtau",
                var="p",
            ),
            make_jac_key(
                k=1,
                owner_type="total_joint",
                owner_name="robot",
                dtype=DTYPE_JOINT,
                field="q",
                var="p",
            ),
        ]

        p = np.array([1.0, 2.0], dtype=float)
        out = builder.build_state(p, required=required)

        self.assertTrue(
            np.allclose(
                out[make_key(k=0, owner_type="total_joint", owner_name="robot", dtype=DTYPE_DYNAMICS, field="tau")],
                np.array([17.0, 44.0], dtype=float),
            )
        )
        self.assertTrue(
            np.allclose(
                out[make_key(k=0, owner_type="total_joint", owner_name="robot", dtype=DTYPE_DYNAMICS, field="dtau")],
                np.array([18.0, 46.0], dtype=float),
            )
        )
        self.assertTrue(
            np.allclose(
                out[make_key(k=1, owner_type="total_joint", owner_name="robot", dtype=DTYPE_DYNAMICS, field="tau")],
                np.array([52.0, 116.0], dtype=float),
            )
        )
        self.assertTrue(
            np.allclose(
                out[make_key(k=1, owner_type="total_joint", owner_name="robot", dtype=DTYPE_DYNAMICS, field="dtau")],
                np.array([48.0, 106.0], dtype=float),
            )
        )

        self.assertTrue(
            np.allclose(
                out[
                    make_jac_key(
                        k=0,
                        owner_type="total_joint",
                        owner_name="robot",
                        dtype=DTYPE_DYNAMICS,
                        field="tau",
                        var="p",
                    )
                ],
                np.array([[17.0, 0.0], [0.0, 22.0]], dtype=float),
            )
        )
        self.assertTrue(
            np.allclose(
                out[
                    make_jac_key(
                        k=0,
                        owner_type="total_joint",
                        owner_name="robot",
                        dtype=DTYPE_DYNAMICS,
                        field="dtau",
                        var="p",
                    )
                ],
                np.array([[18.0, 0.0], [0.0, 23.0]], dtype=float),
            )
        )
        self.assertTrue(
            np.allclose(
                out[
                    make_jac_key(
                        k=1,
                        owner_type="total_joint",
                        owner_name="robot",
                        dtype=DTYPE_DYNAMICS,
                        field="tau",
                        var="p",
                    )
                ],
                np.array([[52.0, 0.0], [0.0, 58.0]], dtype=float),
            )
        )
        self.assertTrue(
            np.allclose(
                out[
                    make_jac_key(
                        k=1,
                        owner_type="total_joint",
                        owner_name="robot",
                        dtype=DTYPE_DYNAMICS,
                        field="dtau",
                        var="p",
                    )
                ],
                np.array([[48.0, 0.0], [0.0, 53.0]], dtype=float),
            )
        )

        self.assertTrue(
            np.allclose(
                out[
                    make_jac_key(
                        k=0,
                        owner_type="total_joint",
                        owner_name="robot",
                        dtype=DTYPE_JOINT,
                        field="q",
                        var="p",
                    )
                ],
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
            )
        )
        self.assertTrue(
            np.allclose(
                out[
                    make_jac_key(
                        k=1,
                        owner_type="total_joint",
                        owner_name="robot",
                        dtype=DTYPE_JOINT,
                        field="q",
                        var="p",
                    )
                ],
                np.array([[6.0, 0.0], [0.0, 7.0]], dtype=float),
            )
        )

        self.assertEqual(model.kinematics_calls, 2)
        self.assertEqual(model.dynamics_calls, 2)

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
            dynamics_fields=("tau",),
        )

        key_tau = make_key(
            k=0,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_DYNAMICS,
            field="tau",
        )
        key_tau_jac = make_jac_key(
            k=0,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_DYNAMICS,
            field="tau",
            var="p",
        )

        out = builder.build_state(np.array([1.0, 2.0], dtype=float), required=[key_tau, key_tau_jac])
        self.assertTrue(np.allclose(out[key_tau], np.array([17.0, 44.0], dtype=float)))
        self.assertTrue(np.allclose(out[key_tau_jac], np.array([[17.0, 0.0], [0.0, 22.0]], dtype=float)))
