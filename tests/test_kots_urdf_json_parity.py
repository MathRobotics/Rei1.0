from __future__ import annotations

import copy
import pytest

from pathlib import Path

import numpy as np

from rei.optimize_backends.kots import compile_kots_trajectory_problem
from rei.optimize_backends.trajectory_ioc import compile_trajectory_ioc_problem, estimate_ioc_weights

try:
    from robokots.kots import Kots
except ImportError:  # pragma: no cover
    Kots = None

_MODEL_CASES = (
    ("planar2", 2),
    ("sample_robot", 3),
    ("7_dof_arm", 7),
)


def _minimal_kots_trajectory_dsl(q_dim: int) -> dict:
    p_dim = 2 * int(q_dim)
    return {
        "time": {"N": 1, "dt": 0.2},
        "trajectory": {
            "type": "linear",
            "var": "p",
            "steps": 2,
            "q_dim": q_dim,
            "A": np.eye(p_dim, dtype=float).tolist(),
        },
        "variables": [{"name": "p", "dim": p_dim, "init": [0.0] * p_dim}],
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
                    "b": {
                        "type": "const",
                        "var": "p",
                        "value": [0.05 * float(i + 1) for i in range(q_dim)],
                    },
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
                    "b": {
                        "type": "const",
                        "var": "p",
                        "value": [-0.04 * float(i + 1) for i in range(q_dim)],
                    },
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
    @pytest.mark.parametrize(("model_name", "q_dim"), _MODEL_CASES)
    def test_kots_urdf_and_json_models_match_runtime_linearization(
        self,
        model_name: str,
        q_dim: int,
    ) -> None:
        if Kots is None:
            pytest.skip("RoboKots is not installed.")
        if not hasattr(Kots, "from_urdf_file"):
            pytest.skip("RoboKots does not expose Kots.from_urdf_file yet.")

        root = Path(__file__).resolve().parents[1]
        json_path = root / "examples" / "models" / f"{model_name}.json"
        urdf_path = root / "examples" / "models" / f"{model_name}.urdf"
        assert json_path.is_file(), f"model not found: {json_path}"
        assert urdf_path.is_file(), f"model not found: {urdf_path}"

        order = 5
        dsl = _minimal_kots_trajectory_dsl(q_dim)

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


def test_kots_batched_trajectory_dynamics_matches_stepwise() -> None:
    if Kots is None:
        pytest.skip("RoboKots is not installed.")

    root = Path(__file__).resolve().parents[1]
    model_path = root / "examples" / "models" / "planar2.json"
    dsl = _minimal_kots_trajectory_dsl(2)
    torque_at_step_one = copy.deepcopy(dsl["terms"][2])
    torque_at_step_one["expr"]["name"] = "tau1"
    torque_at_step_one["expr"]["key"]["k"] = 1
    dsl["terms"].append(torque_at_step_one)

    def linearize(*, batch_trajectory: bool) -> tuple[np.ndarray, np.ndarray]:
        model = Kots.from_json_file(str(model_path), order=5)
        compiled = compile_kots_trajectory_problem(
            dsl,
            model=model,
            data=model.state_dict_,
            kots_backend="rust",
            batch_trajectory=batch_trajectory,
        )
        return compiled.runtime.linearize()

    r_batched, J_batched = linearize(batch_trajectory=True)
    r_stepwise, J_stepwise = linearize(batch_trajectory=False)
    np.testing.assert_allclose(r_batched, r_stepwise, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(J_batched, J_stepwise, rtol=0.0, atol=0.0)


def test_kots_batched_ioc_state_vjp_matches_stepwise() -> None:
    """IOC stacks trajectory state VJPs into one RoboKots batch request."""
    if Kots is None:
        pytest.skip("RoboKots is not installed.")

    class _VjpProbe:
        def __init__(self, model) -> None:
            self._model = model
            self.vjp_rhs_shapes: list[tuple[int, ...]] = []

        def __getattr__(self, name):
            return getattr(self._model, name)

        def jacobian_transpose_mul(self, state_ref, rhs):
            self.vjp_rhs_shapes.append(tuple(np.asarray(rhs).shape))
            return self._model.jacobian_transpose_mul(state_ref, rhs)

    root = Path(__file__).resolve().parents[1]
    model_path = root / "examples" / "models" / "planar2.json"
    dsl = _minimal_kots_trajectory_dsl(2)
    dsl["variables"][0]["init"] = [0.1, -0.2, 0.3, 0.4]
    torque_at_zero = copy.deepcopy(dsl["terms"][2]["expr"])
    torque_at_one = copy.deepcopy(torque_at_zero)
    torque_at_one["name"] = "tau1"
    torque_at_one["key"]["k"] = 1
    dsl["terms"] = [
        {
            "expr": {
                "type": "vstack",
                "name": "torque_stack",
                "parts": [torque_at_zero, torque_at_one],
            },
            "cost": {"type": "l2"},
        }
    ]

    def estimate(*, batch_trajectory: bool) -> tuple[dict, list[tuple[int, ...]]]:
        model = _VjpProbe(Kots.from_json_file(str(model_path), order=5))
        compiled = compile_trajectory_ioc_problem(
            dsl,
            backend="kots",
            model=model,
            data=model.state_dict_,
            kots_backend="rust",
            batch_trajectory=batch_trajectory,
        )
        return estimate_ioc_weights(compiled), model.vjp_rhs_shapes

    stepwise, stepwise_shapes = estimate(batch_trajectory=False)
    batched, batched_shapes = estimate(batch_trajectory=True)

    # Same stationarity result, while the two per-step VJPs become one batch.
    np.testing.assert_allclose(batched["weights"], stepwise["weights"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        batched["stationarity"]["ikkt_residual"],
        stepwise["stationarity"]["ikkt_residual"],
        rtol=0.0,
        atol=0.0,
    )
    assert stepwise_shapes == [(2,), (2,)]
    assert batched_shapes == [(2, 2)]
