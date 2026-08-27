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


@pytest.mark.parametrize("gravity", [None, (0.0, 0.0, -9.81)])
@pytest.mark.parametrize("fields", [("torque",), ("torque_d1",), ("torque", "torque_d1")])
def test_kots_batched_ioc_state_vjp_matches_stepwise(
    fields: tuple[str, ...],
    gravity: tuple[float, float, float] | None,
) -> None:
    """IOC stacks trajectory state VJPs into one RoboKots batch request."""
    if Kots is None:
        pytest.skip("RoboKots is not installed.")

    class _VjpProbe:
        def __init__(self, model) -> None:
            self._model = model
            self.vjp_rhs_shapes: list[tuple[int, ...]] = []
            self.batch_imports = 0
            self.dynamics_updates = 0

        def __getattr__(self, name):
            return getattr(self._model, name)

        def jacobian_transpose_mul(self, state_ref, rhs):
            self.vjp_rhs_shapes.append(tuple(np.asarray(rhs).shape))
            return self._model.jacobian_transpose_mul(state_ref, rhs)

        def import_motions(self, motion) -> None:
            if np.asarray(motion).ndim >= 2:
                self.batch_imports += 1
            self._model.import_motions(motion)

        def dynamics(self, *args, **kwargs):
            self.dynamics_updates += 1
            return self._model.dynamics(*args, **kwargs)

    root = Path(__file__).resolve().parents[1]
    model_path = root / "examples" / "models" / "planar2.json"
    dsl = _minimal_kots_trajectory_dsl(2)
    dsl["variables"][0]["init"] = [0.1, -0.2, 0.3, 0.4]
    state_template = copy.deepcopy(dsl["terms"][2]["expr"])
    dsl["terms"] = []
    for field in fields:
        field_at_zero = copy.deepcopy(state_template)
        field_at_zero["name"] = f"{field}0"
        field_at_zero["key"]["field"] = field
        field_at_one = copy.deepcopy(field_at_zero)
        field_at_one["name"] = f"{field}1"
        field_at_one["key"]["k"] = 1
        dsl["terms"].append(
            {
                "expr": {
                    "type": "vstack",
                    "name": f"{field}_stack",
                    "parts": [field_at_zero, field_at_one],
                },
                "cost": {"type": "l2"},
            }
        )

    def estimate(*, batch_trajectory: bool) -> tuple[dict, list[tuple[int, ...]], int, int, object]:
        model = _VjpProbe(Kots.from_json_file(str(model_path), order=5))
        compiled = compile_trajectory_ioc_problem(
            dsl,
            backend="kots",
            model=model,
            data=model.state_dict_,
            kots_backend="rust",
            batch_trajectory=batch_trajectory,
            gravity=gravity,
        )
        result = estimate_ioc_weights(compiled)
        return result, model.vjp_rhs_shapes, model.batch_imports, model.dynamics_updates, compiled

    stepwise, stepwise_shapes, stepwise_imports, stepwise_updates, _stepwise_compiled = estimate(
        batch_trajectory=False
    )
    batched, batched_shapes, batched_imports, batched_updates, batched_compiled = estimate(batch_trajectory=True)

    # Same stationarity result, while the two per-step VJPs become one batch.
    np.testing.assert_allclose(batched["weights"], stepwise["weights"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        batched["stationarity"]["ikkt_residual"],
        stepwise["stationarity"]["ikkt_residual"],
        rtol=0.0,
        atol=0.0,
    )
    assert stepwise_shapes == [(2,)] * (2 * len(fields))
    assert batched_shapes == [(2, 2)] * len(fields)
    # Value evaluation and the following VJP share the same batch outward state.
    assert batched_imports == 1
    assert batched_updates == 1
    assert stepwise_imports == 0

    # Reuse at the same point, then invalidate on each cache-key input.
    builder = batched_compiled.runtime.ctx.state.build_state.__self__
    p = batched_compiled.runtime.pack.get()
    required = batched_compiled.runtime.required
    builder.build_state(p, time=batched_compiled.runtime.time, required=required)
    assert builder._batched_dynamics_cache_misses == 1
    batched_compiled.runtime.time.update(dt=0.1)
    builder.build_state(p, time=batched_compiled.runtime.time, required=required)
    assert builder._batched_dynamics_cache_misses == 2
    builder.gravity = (0.0, 0.0, 0.0) if gravity is not None else (0.0, 0.0, -9.81)
    builder.build_state(p, time=batched_compiled.runtime.time, required=required)
    assert builder._batched_dynamics_cache_misses == 3
    builder.build_state(np.asarray(p, dtype=float) + 1e-6, time=batched_compiled.runtime.time, required=required)
    assert builder._batched_dynamics_cache_misses == 4


def test_kots_multi_vjp_combines_torque_fields() -> None:
    """Prefer RoboKots' heterogeneous-state VJP API over field grouping."""
    if Kots is None:
        pytest.skip("RoboKots is not installed.")

    class _VjpProbe:
        def __init__(self, model, *, expose_many: bool) -> None:
            self._model = model
            self.expose_many = expose_many
            self.single_calls = 0
            self.many_calls = 0

        def __getattr__(self, name):
            if name == "jacobian_transpose_mul_many" and not self.expose_many:
                raise AttributeError(name)
            return getattr(self._model, name)

        def jacobian_transpose_mul(self, state_ref, rhs):
            self.single_calls += 1
            return self._model.jacobian_transpose_mul(state_ref, rhs)

        def jacobian_transpose_mul_many(self, requests):
            if not self.expose_many:
                raise AttributeError("jacobian_transpose_mul_many")
            self.many_calls += 1
            return self._model.jacobian_transpose_mul_many(requests)

    root = Path(__file__).resolve().parents[1]
    model_path = root / "examples" / "models" / "planar2.json"
    dsl = _minimal_kots_trajectory_dsl(2)
    dsl["variables"][0]["init"] = [0.1, -0.2, 0.3, 0.4]
    template = dsl["terms"][2]["expr"]
    parts = []
    for field in ("torque", "torque_d1"):
        for k in (0, 1):
            part = copy.deepcopy(template)
            part["name"] = f"{field}{k}"
            part["key"]["field"] = field
            part["key"]["k"] = k
            parts.append(part)
    dsl["terms"] = [{"expr": {"type": "vstack", "name": "mixed_tau", "parts": parts}, "cost": {"type": "l2"}}]

    def estimate(*, expose_many: bool) -> tuple[dict, _VjpProbe]:
        probe = _VjpProbe(Kots.from_json_file(str(model_path), order=5), expose_many=expose_many)
        compiled = compile_trajectory_ioc_problem(
            dsl,
            backend="kots",
            model=probe,
            data=probe.state_dict_,
            kots_backend="rust",
            batch_trajectory=True,
        )
        return estimate_ioc_weights(compiled), probe

    grouped, grouped_probe = estimate(expose_many=False)
    multi, multi_probe = estimate(expose_many=True)
    np.testing.assert_allclose(multi["weights"], grouped["weights"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        multi["stationarity"]["ikkt_residual"],
        grouped["stationarity"]["ikkt_residual"],
        rtol=0.0,
        atol=1e-10,
    )
    assert grouped_probe.single_calls == 2
    assert multi_probe.many_calls == 1
    assert multi_probe.single_calls == 0
