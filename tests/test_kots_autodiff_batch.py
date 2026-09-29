"""Real JIT AD through trajectory, DOC and IOC products, without analytic AD fallback."""
from pathlib import Path

import numpy as np
import pytest

from rei import load_problem_spec_toml, solve
from rei.optimize.reductions import build_nullspace_equality_reduction
from rei.optimize_backends.kots import compile_kots_trajectory_problem
from rei.optimize_backends.trajectory_ioc import estimate_ioc_weights


ROOT = Path(__file__).resolve().parents[1]


def compile_problem(method="autodiff", batch=True, backend="numpy"):
    Kots = pytest.importorskip("robokots.kots").Kots
    pytest.importorskip("jax")
    spec = load_problem_spec_toml(ROOT / "examples/spec/robokots_traj_dynamics_d12.toml")
    spec["time"].update(N=6, dt=0.2)
    spec["trajectory"]["num_ctrl_points"] = 6
    model = Kots.from_urdf_file(str(ROOT / "examples/models/planar2.urdf"), order=1, backend=backend)
    return compile_kots_trajectory_problem(
        spec, model=model, jacobian_method=method, batch_trajectory=batch,
        kots_backend=backend, gravity=(0., -9.81, 0.), max_derivative_order=4,
    )


def forbid_analytic(monkeypatch, model):
    def forbidden(*args, **kwargs):
        raise AssertionError("Autodiff must not call analytic derivative APIs")
    for name in ("jacobian", "jacobian_mul", "jacobian_transpose_mul", "jacobian_transpose_mul_many",
                 "squared_power_torque_vjp_terms"):
        if hasattr(model, name):
            monkeypatch.setattr(model, name, forbidden)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_batch_doc_ioc_products_and_mutations(monkeypatch, backend):
    batch = compile_problem(backend=backend)
    single = compile_problem(batch=False, backend=backend)
    analytic = compile_problem(method="analytic", backend=backend)
    builder = batch.state_builder
    forbid_analytic(monkeypatch, builder.model)
    calls = []
    native = builder.model.jacobian_autodiff

    def watched(refs, **kwargs):
        result = native(refs, **kwargs)
        calls.append((len(refs) if isinstance(refs, list) else 1, result.shape))
        assert kwargs["jit"]
        return result

    monkeypatch.setattr(builder.model, "jacobian_autodiff", watched)
    rng = np.random.default_rng(73)
    for scale, gravity in ((1e-12, (0., -9.81, 0.)), (0.2, (0., 0., 0.))):
        point = rng.normal(scale=scale, size=batch.runtime.pack.n_total)
        results = []
        for compiled in (batch, single, analytic):
            compiled.state_builder.gravity = gravity
            compiled.runtime.pack.set(point)
            compiled.runtime.state.invalidate()
            results.append(compiled.runtime.linearize())
        r, J = results[0]
        for ref_r, ref_J in results[1:]:
            np.testing.assert_allclose(r, ref_r, atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(J, ref_J, atol=1e-8, rtol=1e-8)
        v, w = rng.normal(size=J.shape[1]), rng.normal(size=J.shape[0])
        np.testing.assert_allclose(batch.runtime.residual_jvp(v, weighted=True), J @ v, atol=1e-8)
        np.testing.assert_allclose(batch.runtime.residual_vjp(w, weighted=True), J.T @ w, atol=1e-8)
        *_, gradients = batch.runtime.term_gradient_contributions()
        *_, expected = analytic.runtime.term_gradient_contributions()
        np.testing.assert_allclose(gradients, expected, atol=1e-8, rtol=1e-8)
    assert any(count == 2 and len(shape) == 3 and shape[0] == 7 and shape[-1] == 6
               for count, shape in calls), calls

    keys = [key for key in batch.runtime.operator_required_list() if key.dtype == "dynamics"]
    selected = [keys[-1], keys[1], keys[-1]]
    requests = [(key, rng.normal(size=2)) for key in selected]
    actual = builder.param_jacobian_transpose_mul_many(point, requests, pack=batch.runtime.pack,
                                                     time=batch.runtime.time)
    expected = [single.state_builder.param_jacobian_transpose_mul(
        point, key, rhs, pack=single.runtime.pack, time=single.runtime.time,
    ) for key, rhs in requests]
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    columns = builder.param_jacobian_transpose_mul_many_fused_columns(
        point, [requests[:2], requests[2:]], pack=batch.runtime.pack, time=batch.runtime.time,
    )
    np.testing.assert_allclose(columns, np.stack([sum(expected[:2]), expected[2]], axis=1), atol=1e-8)
    matrix_requests = [(key, np.stack([rhs, -2 * rhs], axis=1)) for key, rhs in requests]
    matrix_products = builder.param_jacobian_transpose_mul_many(
        point, matrix_requests, pack=batch.runtime.pack, time=batch.runtime.time,
    )
    np.testing.assert_allclose(matrix_products, np.stack([expected, -2 * np.asarray(expected)], axis=-1), atol=1e-8)
    # An intervening scalar evaluation must not leave a stale batch in use.
    builder.param_jacobian_transpose_mul(point, *requests[0], pack=batch.runtime.pack, time=batch.runtime.time)
    again = builder.param_jacobian_transpose_mul_many(point, requests, pack=batch.runtime.pack, time=batch.runtime.time)
    np.testing.assert_allclose(again, expected, atol=1e-8)
    builder.model.set_order(6)
    changed = builder.param_jacobian_transpose_mul_many(point, requests, pack=batch.runtime.pack, time=batch.runtime.time)
    np.testing.assert_allclose(changed, expected, atol=1e-8)


def test_output_and_model_changes_are_not_cached(monkeypatch):
    from dataclasses import replace

    rng = np.random.default_rng(91)
    # Register both fields, then alternate output selections at the same point.
    # A fresh compile is needed for changing the registered field set.
    def expanded(method):
        Kots = pytest.importorskip("robokots.kots").Kots
        spec = load_problem_spec_toml(ROOT / "examples/spec/robokots_traj_dynamics_d12.toml")
        spec["time"].update(N=6, dt=.2)
        spec["trajectory"]["num_ctrl_points"] = 6
        return compile_kots_trajectory_problem(
            spec, model=Kots.from_urdf_file(str(ROOT / "examples/models/planar2.urdf")),
            jacobian_method=method, dynamics_fields=["torque", "torque_d1", "force", "momentum"],
            gravity=(0., -9.81, 0.), max_derivative_order=4,
        )
    compiled, reference = expanded("autodiff"), expanded("analytic")
    point = rng.normal(scale=.1, size=compiled.runtime.pack.n_total)
    keys = [key for key in compiled.runtime.operator_required_list() if key.dtype == "dynamics"]
    directions = {key.k: rng.normal(size=compiled.trajectory_map.p_dim) for key in keys}
    builder = compiled.state_builder
    forbid_analytic(monkeypatch, builder.model)
    for mass_factor in (1., 1.3):
        for target in (compiled, reference):
            target.state_builder.model.robot_.links[-1].mass *= mass_factor
        for fields in (("torque", "torque_d1"), ("force", "momentum"), ("torque",)):
            requests = [(replace(key, field=field), directions[key.k])
                        for field in fields for key in (keys[-1], keys[0])]
            actual = builder.param_jacobian_mul_many(point, requests, pack=compiled.runtime.pack,
                                                     time=compiled.runtime.time)
            # Force a fresh analytic reference after direct physical-model edits.
            reference.state_builder._batched_dynamics_cache_key = None
            jac_keys = [replace(key, field=key.field + "_J_p") for key, _ in requests]
            matrices = reference.state_builder.build_state(
                point, required=jac_keys, pack=reference.runtime.pack, time=reference.runtime.time,
            )
            expected = [matrices[key] @ direction
                        for key, (_, direction) in zip(jac_keys, requests, strict=True)]
            np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)


def test_doc_to_ioc_completes_without_analytic(monkeypatch):
    compiled = compile_problem()
    forbid_analytic(monkeypatch, compiled.state_builder.model)
    reduction = build_nullspace_equality_reduction(
        compiled.runtime, eq_selector_attr="enforce", eq_selector_value="nullspace",
    )
    outcome = solve(reduction.runtime, options={"verbose": False, "tol_grad": 1e-8, "max_iters": 100})
    assert outcome.converged
    result = estimate_ioc_weights(compiled, p=reduction.lift(outcome.solution))
    assert np.all(np.isfinite(result["weights"]))
    assert np.isfinite(result["stationarity"]["ikkt_residual_norm"])
