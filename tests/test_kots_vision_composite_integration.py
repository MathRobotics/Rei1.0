from __future__ import annotations

import importlib
import sys
import types
import unittest
from typing import Any

import numpy as np

from eiopt.backends.state.composite import CompositeStateBuilder
from eiopt.backends.state.vision import CameraCalibrationStateProvider, VisionFieldHandler
from eiopt.core.state_schema import DTYPE_KINEMATICS, make_key
from eiopt.optimize.builder import compile_nls_problem
from eiopt.optimize.solvers import solve


def _ensure_robokots_state_stub() -> None:
    robokots_mod = types.ModuleType("robokots")
    core_mod = types.ModuleType("robokots.core")
    state_mod = types.ModuleType("robokots.core.state")

    class StateType:  # noqa: D401
        """Minimal stub for RoboKots StateType."""

        def __init__(self, owner_type: str, owner_name: str, field: str, frame: str | None) -> None:
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
if "eiopt.backends.state.kots" in sys.modules:
    _kots_mod = importlib.reload(sys.modules["eiopt.backends.state.kots"])
else:
    _kots_mod = importlib.import_module("eiopt.backends.state.kots")
KotsStateBuilder = _kots_mod.KotsStateBuilder


class _FakeKotsModel:
    def __init__(self) -> None:
        self._motion = np.zeros((2,), dtype=float)
        self.kinematics_calls = 0

    def dof(self) -> int:
        return 2

    def order(self) -> int:
        return 1

    def import_motions(self, motion: np.ndarray) -> None:
        self._motion = np.asarray(motion, dtype=float).reshape(-1).copy()

    def kinematics(self) -> None:
        self.kinematics_calls += 1

    @staticmethod
    def _state_field_name(state_ref: Any) -> str | None:
        for attr in ("field", "field_", "data_type", "dtype"):
            value = getattr(state_ref, attr, None)
            if isinstance(value, str) and value != "":
                return value
        return None

    def state_info(self, state_ref: Any) -> np.ndarray:
        field = self._state_field_name(state_ref)
        if field == "pos":
            q0 = float(self._motion[0])
            q1 = float(self._motion[1])
            return np.array([q0, q1, 0.0], dtype=float)
        raise ValueError(f"Unsupported field: {field!r}")

    def jacobian(self, state_ref: Any) -> np.ndarray:
        field = self._state_field_name(state_ref)
        if field == "pos":
            return np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ],
                dtype=float,
            )
        raise ValueError(f"Unsupported field: {field!r}")


class TestKotsVisionCompositeIntegration(unittest.TestCase):
    def test_composite_state_builder_solves_joint_robot_camera_problem(self) -> None:
        model = _FakeKotsModel()
        kots_builder = KotsStateBuilder(
            model,
            data={},
            q_var="q",
            fields=("pos",),
            dynamics_fields=None,
        )

        H = np.array(
            [
                [1.5, -0.4],
                [-0.7, 1.2],
            ],
            dtype=float,
        )
        b = np.array([0.2, -0.1], dtype=float)

        def _vision_value(q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
            del key, state_ref
            q_vec = np.asarray(q, dtype=float).reshape(-1)
            if q_vec.size != 2:
                raise ValueError(f"vision value: expected q size 2, got {q_vec.size}.")
            return H @ q_vec + b

        def _vision_jac(q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
            del q, key, state_ref
            return H.copy()

        vision_provider = CameraCalibrationStateProvider(
            model={"name": "cam0"},
            data={},
            param_var="q",
            owner_type="camera",
            field_handlers={
                "reproj": VisionFieldHandler(
                    value_handler=_vision_value,
                    jac_handler=_vision_jac,
                )
            },
        )

        composite = CompositeStateBuilder([kots_builder, vision_provider])

        q_true = np.array([0.8, -0.35], dtype=float)
        robot_key = make_key(
            k=0,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
        )
        target_pos = np.asarray(kots_builder.build_state(q_true, required=[robot_key])[robot_key], dtype=float)
        observations = H @ q_true + b

        dsl = {
            "variables": [
                {
                    "name": "q",
                    "dim": 2,
                    "init": [0.0, 0.0],
                }
            ],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "name": "robot_pos_error",
                        "a": {
                            "type": "get_state",
                            "key": {
                                "k": 0,
                                "owner_type": "link",
                                "owner_name": "ee",
                                "dtype": "kinematics",
                                "field": "pos",
                                "frame": "world",
                            },
                            "jac": {"var": "q"},
                        },
                        "b": {
                            "type": "const",
                            "var": "q",
                            "value": target_pos.tolist(),
                        },
                    },
                    "cost": {"type": "l2"},
                },
                {
                    "expr": {
                        "type": "sub",
                        "name": "camera_reproj_error",
                        "a": {
                            "type": "get_state",
                            "key": {
                                "k": 0,
                                "owner_type": "camera",
                                "owner_name": "cam0",
                                "dtype": "vision",
                                "field": "reproj",
                            },
                            "jac": {"var": "q"},
                        },
                        "b": {
                            "type": "const",
                            "var": "q",
                            "value": observations.tolist(),
                        },
                    },
                    "cost": {"type": "l2"},
                },
            ],
        }

        runtime = compile_nls_problem(dsl, build_state=composite.build_state)
        x_star, _cost0, cost, _iters, _rnorm, _dxnorm, converged = solve(
            runtime,
            solver="gauss_newton",
            max_iters=20,
            gn_damping=0.0,
            gn_line_search=False,
        )

        self.assertTrue(converged)
        self.assertLess(cost, 1e-20)
        np.testing.assert_allclose(x_star, q_true, rtol=0.0, atol=1e-10)
        self.assertGreater(model.kinematics_calls, 0)


if __name__ == "__main__":
    unittest.main()
