from __future__ import annotations

import importlib
import sys
import types

import numpy as np

from rei.core.state_schema import DTYPE_COORD

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

    def computeGeneralizedGravity(model, data, q):
        del model, data, q
        return np.zeros((1,), dtype=float)

    def rnea(model, data, q, v, a):
        del model, data, q, v
        return 3.0 * np.asarray(a, dtype=float).reshape(-1)

    def computeRNEADerivatives(model, data, q, v, a):
        del model, data, q, v, a
        return (
            np.array([[2.0]], dtype=float),
            np.array([[5.0]], dtype=float),
            np.array([[3.0]], dtype=float),
        )

    pin.ReferenceFrame = _ReferenceFrame
    pin.buildModelFromUrdf = buildModelFromUrdf
    pin.computeFrameJacobian = computeFrameJacobian
    pin.forwardKinematics = forwardKinematics
    pin.updateFramePlacements = updateFramePlacements
    pin.computeGeneralizedGravity = computeGeneralizedGravity
    pin.rnea = rnea
    pin.computeRNEADerivatives = computeRNEADerivatives

    sys.modules["pinocchio"] = pin

_ensure_pinocchio_stub()
_pin_opt_mod = importlib.import_module("rei.optimize_backends.pinocchio")
compile_pinocchio_trajectory_problem = _pin_opt_mod.compile_pinocchio_trajectory_problem
from rei.optimize_backends.trajectory_diagnostics import inspect_trajectory_problem_backend
from rei.optimize_backends.trajectory_ioc import compile_trajectory_ioc_problem, estimate_ioc_weights

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

class TestPinocchioTrajectoryCompileMock:
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

        assert compiled.p_var == "p"
        assert compiled.trajectory_map.p_dim == 2
        assert compiled.trajectory_map.steps == 2
        assert compiled.runtime.pack.n_total == 2

        r, J = compiled.runtime.linearize()
        assert np.allclose(r, np.array([-1.0, -2.0], dtype=float))
        assert np.allclose(J, np.eye(2, dtype=float))

    def test_compile_pinocchio_trajectory_problem_torque_uses_trajectory_dynamics(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.5},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 3,
                "q_dim": 1,
                "A": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "variables": [
                {"name": "p", "dim": 3, "init": [0.0, 0.0, 0.0]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "name": "torque_mid",
                        "key": {
                            "k": 1,
                            "owner_type": "total_joint",
                            "owner_name": "robot",
                            "dtype": "dynamics",
                            "field": "torque",
                        },
                        "jac": {"var": "p"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        compiled = compile_pinocchio_trajectory_problem(
            dsl,
            model=_FakePinModel(),
            data=_FakePinData(),
            fields=("pos",),
        )
        assert compiled.model_order == 3
        assert sorted(compiled.trajectory_derivative_maps.keys()) == [0, 1, 2]

        runtime = compiled.runtime
        x_target = np.array([0.0, 1.0, 0.0], dtype=float)
        runtime.pack.apply_dx(x_target - runtime.pack.get())

        terms = runtime.linearize_terms(weighted=False)
        assert len(terms) == 1
        r = terms[0].residual
        J = terms[0].jacobian
        assert not (np.allclose(r, np.zeros_like(r)))
        assert not (np.allclose(J, np.zeros_like(J)))

    def test_compile_pinocchio_trajectory_problem_torque_uses_analytic_derivatives(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.5},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 3,
                "q_dim": 1,
                "A": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "variables": [
                {"name": "p", "dim": 3, "init": [0.0, 1.0, 0.0]},
            ],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "name": "torque_mid",
                        "key": {
                            "k": 1,
                            "owner_type": "total_joint",
                            "owner_name": "robot",
                            "dtype": "dynamics",
                            "field": "torque",
                        },
                        "jac": {"var": "p"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        compiled = compile_pinocchio_trajectory_problem(
            dsl,
            model=_FakePinModel(),
            data=_FakePinData(),
            fields=("pos",),
            torque_jacobian="analytic",
        )

        terms = compiled.runtime.linearize_terms(weighted=False)
        J = terms[0].jacobian
        k = 1
        J_ref = (
            2.0 * compiled.trajectory_derivative_maps[0].dqdp_at(k)
            + 5.0 * compiled.trajectory_derivative_maps[1].dqdp_at(k)
            + 3.0 * compiled.trajectory_derivative_maps[2].dqdp_at(k)
        )
        assert np.allclose(J, J_ref)

    def test_compile_pinocchio_trajectory_problem_warn_skip_unsupported_torque_d1(self) -> None:
        dsl = {
            "time": {"N": 1, "dt": 0.5},
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
            "variables": [{"name": "p", "dim": 2, "init": [0.0, 0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "get_state",
                        "name": "torque_d1_term",
                        "key": {
                            "k": 0,
                            "owner_type": "total_joint",
                            "owner_name": "robot",
                            "dtype": "dynamics",
                            "field": "torque_d1",
                        },
                        "jac": {"var": "p"},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        compiled = compile_pinocchio_trajectory_problem(
            dsl,
            model=_FakePinModel(),
            data=_FakePinData(),
            fields=("pos",),
            unsupported="warn_skip",
            max_derivative_order=3,
        )

        assert compiled.runtime.linearize_terms(weighted=False) == []
        assert compiled.diagnostics is not None
        assert len(compiled.diagnostics.warnings) == 1
        warning = compiled.diagnostics.warnings[0]
        assert warning.field == "torque_d1"
        assert warning.action == "skipped"

    def test_inspect_trajectory_problem_backend_reports_low_derivative_order(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.5},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 3,
                "q_dim": 1,
                "A": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "variables": [{"name": "p", "dim": 3, "init": [0.0, 1.0, 0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "get_traj_var",
                        "name": "qdddot_regularization",
                        "var": "p",
                        "derivative_order": 3,
                        "derivative_wrt": "time",
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }

        report = inspect_trajectory_problem_backend(
            dsl,
            backend="pinocchio",
            model=_FakePinModel(),
            data=_FakePinData(),
            max_derivative_order=2,
            unsupported_action="skipped",
        )

        assert report.backend == "pinocchio"
        assert len(report.unsupported_terms) == 1
        diag = report.unsupported_terms[0]
        assert diag.term_name == "qdddot_regularization"
        assert diag.dtype == "trajectory"
        assert diag.field == "qdddot"
        assert "max_derivative_order=2" in diag.reason
        payload = report.to_json_dict()
        assert payload["unsupported_terms"][0]["field"] == "qdddot"

    def test_compile_pinocchio_trajectory_problem_warn_skip_low_derivative_order(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.5},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 3,
                "q_dim": 1,
                "A": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "variables": [{"name": "p", "dim": 3, "init": [0.0, 1.0, 0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "get_traj_var",
                        "name": "qddot_regularization",
                        "var": "p",
                        "derivative_order": 2,
                        "derivative_wrt": "time",
                    },
                    "cost": {"type": "l2"},
                },
                {
                    "expr": {
                        "type": "get_traj_var",
                        "name": "qdddot_regularization",
                        "var": "p",
                        "derivative_order": 3,
                        "derivative_wrt": "time",
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
            max_derivative_order=2,
            unsupported="warn_skip",
        )

        terms = compiled.runtime.linearize_terms(weighted=False)
        assert [t.name for t in terms] == ["qddot_regularization"]
        assert compiled.diagnostics is not None
        assert [d.field for d in compiled.diagnostics.warnings] == ["qdddot"]

    def test_compile_trajectory_ioc_problem_and_estimate_schema(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.5},
            "trajectory": {
                "type": "linear",
                "var": "p",
                "steps": 3,
                "q_dim": 1,
                "A": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "variables": [{"name": "p", "dim": 3, "init": [0.0, 1.0, 0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "get_traj_var",
                        "name": "qdot_regularization",
                        "var": "p",
                        "derivative_order": 1,
                        "derivative_wrt": "time",
                    },
                    "cost": {"type": "l2"},
                },
                {
                    "expr": {
                        "type": "get_state",
                        "name": "torque_d1_term",
                        "key": {
                            "k": 1,
                            "owner_type": "total_joint",
                            "owner_name": "robot",
                            "dtype": "dynamics",
                            "field": "torque_d1",
                        },
                        "jac": {"var": "p"},
                    },
                    "cost": {"type": "l2"},
                },
            ],
        }

        compiled = compile_trajectory_ioc_problem(
            dsl,
            backend="pinocchio",
            model=_FakePinModel(),
            data=_FakePinData(),
            fields=("pos",),
            max_derivative_order=3,
            unsupported="warn_skip",
        )
        result = estimate_ioc_weights(
            compiled,
            p=np.array([0.0, 1.0, 0.0], dtype=float),
            stationarity_scaling="gradient_norm",
        )

        assert result["backend"] == "pinocchio"
        assert len(result["terms"]) == 1
        assert len(result["skipped_terms"]) == 1
        assert "weights" in result
        assert "stationarity_scaling" in result
