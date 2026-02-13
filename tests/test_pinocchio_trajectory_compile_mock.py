from __future__ import annotations

import importlib
import sys
import types
import unittest

import numpy as np

from eiopt.core.state_schema import DTYPE_COORD


def _ensure_pinocchio_stub() -> None:
    pin = types.ModuleType("pinocchio")

    class _ReferenceFrame:
        LOCAL_WORLD_ALIGNED = 0
        LOCAL = 1

    def buildModelFromUrdf(*args, **kwargs):
        del args, kwargs
        return object()

    def computeFrameJacobian(model, data, q, frame_id, rf=None):
        del model, data, q, frame_id, rf
        return np.zeros((6, 1), dtype=float)

    def forwardKinematics(model, data, q):
        del model
        data.last_q = np.asarray(q, dtype=float).reshape(-1).copy()

    def updateFramePlacements(model, data):
        del model, data
        return None

    pin.ReferenceFrame = _ReferenceFrame
    pin.buildModelFromUrdf = buildModelFromUrdf
    pin.computeFrameJacobian = computeFrameJacobian
    pin.forwardKinematics = forwardKinematics
    pin.updateFramePlacements = updateFramePlacements

    sys.modules["pinocchio"] = pin


_ensure_pinocchio_stub()
_pin_mod = importlib.import_module("eiopt.backends.pinocchio")
compile_pinocchio_trajectory_problem = _pin_mod.compile_pinocchio_trajectory_problem


class _FakePinModel:
    nq = 1
    nv = 1

    def getFrameId(self, owner_name: str) -> int:
        del owner_name
        return 0


class _FakePinData:
    def __init__(self) -> None:
        self.last_q = np.zeros((1,), dtype=float)
        frame = types.SimpleNamespace(
            translation=np.zeros((3,), dtype=float),
            rotation=np.eye(3, dtype=float),
        )
        self.oMf = [frame]


class TestPinocchioTrajectoryCompileMock(unittest.TestCase):
    def test_compile_pinocchio_trajectory_problem_for_joint_q_constraints(self) -> None:
        dsl = {
            "time": {"N": 1, "dt": 0.2},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 2,
                "q_dim": 1,
                "A": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            },
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
                        "b": {"type": "const", "var": "p", "dim": 1, "value": [1.0]},
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
                        "b": {"type": "const", "var": "p", "dim": 1, "value": [2.0]},
                    },
                    "cost": {"type": "l2"},
                },
            ],
        }

        compiled = compile_pinocchio_trajectory_problem(
            dsl,
            model=_FakePinModel(),
            data=_FakePinData(),
            fields=("pos",),
        )

        self.assertEqual(compiled.p_var, "p")
        self.assertEqual(compiled.trajectory_map.p_dim, 2)
        self.assertEqual(compiled.trajectory_map.steps, 2)
        self.assertEqual(compiled.runtime.pack.n_total, 2)

        r, J = compiled.runtime.linearize()
        self.assertTrue(np.allclose(r, np.array([-1.0, -2.0], dtype=float)))
        self.assertTrue(np.allclose(J, np.eye(2, dtype=float)))


if __name__ == "__main__":
    unittest.main()
