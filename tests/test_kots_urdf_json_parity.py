from __future__ import annotations

import pytest

from pathlib import Path

import numpy as np

from eiopt.optimize_backends.kots import compile_kots_trajectory_problem

try:
    from robokots.kots import Kots
except ImportError:  # pragma: no cover
    Kots = None

def _minimal_kots_trajectory_dsl() -> dict:
    return {
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
        "variables": [{"name": "p", "dim": 4, "init": [0.0, 0.0, 0.0, 0.0]}],
        "terms": [
            {
                "expr": {
                    "type": "sub",
                    "name": "q0_eq",
                    "a": {
                        "type": "get_state",
                        "key": {
                            "k": 0,
                            "owner_type": "total_joint",
                            "owner_name": "robot",
                            "dtype": "coord",
                            "field": "q",
                        },
                        "jac": {"var": "p"},
                    },
                    "b": {"type": "const", "var": "p", "value": [0.1, -0.2]},
                },
                "cost": {"type": "l2"},
            },
            {
                "expr": {
                    "type": "sub",
                    "name": "q1_eq",
                    "a": {
                        "type": "get_state",
                        "key": {
                            "k": 1,
                            "owner_type": "total_joint",
                            "owner_name": "robot",
                            "dtype": "coord",
                            "field": "q",
                        },
                        "jac": {"var": "p"},
                    },
                    "b": {"type": "const", "var": "p", "value": [0.2, 0.3]},
                },
                "cost": {"type": "l2"},
            },
            {
                "expr": {
                    "type": "get_state",
                    "name": "tau0",
                    "key": {
                        "k": 0,
                        "owner_type": "total_joint",
                        "owner_name": "robot",
                        "dtype": "dynamics",
                        "field": "torque",
                    },
                    "jac": {"var": "p"},
                },
                "cost": {"type": "l2"},
            },
            {
                "expr": {
                    "type": "get_state",
                    "name": "tau_d1_0",
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
            },
            {
                "expr": {
                    "type": "get_state",
                    "name": "tau_d2_0",
                    "key": {
                        "k": 0,
                        "owner_type": "total_joint",
                        "owner_name": "robot",
                        "dtype": "dynamics",
                        "field": "torque_d2",
                    },
                    "jac": {"var": "p"},
                },
                "cost": {"type": "l2"},
            },
        ],
    }

class TestKotsUrdfJsonParity:
    def test_kots_urdf_and_json_models_match_runtime_linearization(self) -> None:
        if Kots is None:
            pytest.skip("RoboKots is not installed.")
        if not hasattr(Kots, "from_urdf_file"):
            pytest.skip("RoboKots does not expose Kots.from_urdf_file yet.")

        root = Path(__file__).resolve().parents[1]
        json_path = root / "examples" / "models" / "planar2.json"
        urdf_path = root / "examples" / "models" / "planar2.urdf"
        assert json_path.is_file(), f"model not found: {json_path}"
        assert urdf_path.is_file(), f"model not found: {urdf_path}"

        order = 5
        dsl = _minimal_kots_trajectory_dsl()

        model_json = Kots.from_json_file(str(json_path), order=order)
        model_urdf = Kots.from_urdf_file(str(urdf_path), order=order)

        compiled_json = compile_kots_trajectory_problem(
            dsl,
            model=model_json,
            data=model_json.state_dict_,
        )
        compiled_urdf = compile_kots_trajectory_problem(
            dsl,
            model=model_urdf,
            data=model_urdf.state_dict_,
        )
        assert compiled_json.model_order == compiled_urdf.model_order
        assert compiled_json.runtime.pack.n_total == compiled_urdf.runtime.pack.n_total

        n_total = int(compiled_json.runtime.pack.n_total)
        rng = np.random.default_rng(0)
        samples = [
            np.zeros((n_total,), dtype=float),
            rng.standard_normal(n_total),
            rng.standard_normal(n_total),
        ]

        for i, x in enumerate(samples):
            compiled_json.runtime.pack.apply_dx(x - compiled_json.runtime.pack.get())
            compiled_urdf.runtime.pack.apply_dx(x - compiled_urdf.runtime.pack.get())
            r_json, J_json = compiled_json.runtime.linearize()
            r_urdf, J_urdf = compiled_urdf.runtime.linearize()
            np.testing.assert_allclose(
                r_json,
                r_urdf,
                rtol=0.0,
                atol=0.0,
                err_msg=f"residual mismatch at sample {i}",
            )
            np.testing.assert_allclose(
                J_json,
                J_urdf,
                rtol=0.0,
                atol=0.0,
                err_msg=f"jacobian mismatch at sample {i}",
            )

