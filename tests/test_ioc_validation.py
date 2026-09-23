"""Known-answer IOC checks, including limitations of objective-only fitting."""

from types import SimpleNamespace

import numpy as np
import pytest

from rei import compile_nls_problem_spec, solve
from rei.optimize.kkt import check_kkt_residuals
from rei.optimize_backends.trajectory_ioc import estimate_ioc_weights


def quadratic_runtime(point, targets, weights=None, constraint=None):
    if weights is None:
        weights = np.ones(len(targets))
    terms = [
        {"name": f"objective_{i}", "var": "x", "target": target, "weight": float(weight)}
        for i, (target, weight) in enumerate(zip(targets, weights))
    ]
    if constraint is not None:
        terms.append(constraint)
    return compile_nls_problem_spec(
        {"opt_vals": {"x": {"init": point}}, "terms": terms},
        build_state=lambda *args, **kwargs: {},
    )


def estimate(runtime, **kwargs):
    return estimate_ioc_weights(SimpleNamespace(runtime=runtime), **kwargs)


@pytest.mark.parametrize("mode", ["none", "gradient_norm", "residual_norm", "weighted_residual_norm"])
def test_known_weights_and_forward_reconstruction(mode):
    targets = [[0.0], [2.0]]
    # The unknown generating weights are deliberately not stored in the IOC runtime.
    result = estimate(quadratic_runtime([0.5], targets), stationarity_scaling=mode)
    np.testing.assert_allclose(result["weights"], [0.75, 0.25], atol=1e-10)
    assert result["stationarity"]["ikkt_residual_norm"] < 1e-10
    assert result["identifiability"]["unique_stationary_weights"]
    recovered = solve(quadratic_runtime([0.0], targets, result["weights"]))
    np.testing.assert_allclose(recovered.solution, [0.5], atol=1e-8)


def test_original_and_scaled_coefficients_have_distinct_meanings():
    result = estimate(quadratic_runtime([0.5], [[0.0], [2.0]]))
    np.testing.assert_allclose(result["scaled_weights"], [0.5, 0.5], atol=1e-10)
    scales = np.asarray(result["stationarity_scaling"]["column_scale"])
    original = scales * result["scaled_weights"]
    np.testing.assert_allclose(result["weights"], original / original.sum())
    np.testing.assert_allclose([t["weight"] for t in result["terms"]], result["weights"])


def test_two_dimensional_identifiable_three_objectives():
    targets = [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]
    truth = np.array([0.2, 0.3, 0.5])
    demo = truth @ np.asarray(targets)
    result = estimate(quadratic_runtime(demo.tolist(), targets))
    np.testing.assert_allclose(result["weights"], truth, atol=1e-9)
    assert result["identifiability"]["active_augmented_rank"] == 3
    assert result["identifiability"]["unique_stationary_weights"]


def test_duplicate_objectives_are_not_identifiable():
    result = estimate(quadratic_runtime([0.5], [[0.0], [0.0], [2.0]]))
    assert result["identifiability"]["active_affine_nullity"] == 1
    assert not result["identifiability"]["unique_stationary_weights"]
    # Many weights generate the same demonstration; only their sum is observable.
    w = result["weights"]
    np.testing.assert_allclose([w[0] + w[1], w[2]], [0.75, 0.25], atol=1e-10)


def test_zero_gradient_terms_are_explicitly_unidentified():
    result = estimate(quadratic_runtime([0.5], [[0.0], [2.0], [0.5]]))
    assert result["identifiability"]["excluded_term_indices"] == [2]
    assert not result["identifiability"]["unique_stationary_weights"]


def test_all_zero_gradients_do_not_claim_a_weight_estimate():
    result = estimate(quadratic_runtime([0.0], [[0.0], [0.0]]))
    assert result["simplex"] is None
    assert not result["identifiability"]["unique_stationary_weights"]
    assert any("No active terms" in m for m in result["validation"]["messages"])


def test_inconsistent_noisy_demonstrations_have_nonzero_residual():
    result = estimate(quadratic_runtime([0.5, 1.1], [[0.0, 0.0], [2.0, 4.0]]))
    assert result["stationarity"]["ikkt_residual_scaled_norm"] > 1e-3
    assert not result["identifiability"]["unique_stationary_weights"]
    assert any("exact scaled stationarity" in m for m in result["validation"]["messages"])


@pytest.mark.parametrize("kind", ["eq", "ineq"])
@pytest.mark.parametrize("include_constraints", [False, True])
def test_constraint_multipliers_cannot_be_replaced_by_objective_weights(kind, include_constraints):
    # The true objective weights [0.1, 0.9] prefer x=1.8. A constraint fixes
    # or caps x at .5. Unconstrained IOC instead infers [.75, .25].
    runtime = quadratic_runtime(
        [0.5], [[0.0], [2.0]],
        constraint={"name": "limit", "var": "x", "target": [0.5], "kind": kind},
    )
    kwargs = {f"{kind}_residual": [0.0], f"{kind}_jacobian": [[1.0]]}
    kkt = check_kkt_residuals(grad_objective=[-1.3], **kwargs)
    assert kkt.ok
    result = estimate(runtime, include_constraints=include_constraints)
    np.testing.assert_allclose(result["weights"][:2], [0.75, 0.25], atol=1e-10)
    assert result["validation"]["constraint_term_indices"] == [2]
    assert not result["validation"]["constraint_kkt_checked"]
    assert not result["identifiability"]["unique_stationary_weights"]
    assert any("KKT multipliers" in m for m in result["validation"]["messages"])


@pytest.mark.parametrize("eps", [0.0, -1.0, np.nan, np.inf])
def test_invalid_scaling_epsilon_is_rejected(eps):
    with pytest.raises(ValueError, match="finite and positive"):
        estimate(quadratic_runtime([0.5], [[0.0], [2.0]]), stationarity_scale_eps=eps)


@pytest.mark.parametrize("factor", [0.01, 100.0, 1e6])
def test_rescaling_one_objective_preserves_its_physical_weight(factor):
    runtime = quadratic_runtime([0.5], [[0.0], [2.0]])
    base = runtime.term_gradient_contributions

    def rescaled(**kwargs):
        indices, names, attrs, raw, weighted, gradients = base(**kwargs)
        raw[0] = raw[0] * np.sqrt(factor)
        weighted[0] = weighted[0] * np.sqrt(factor)
        gradients[0] = gradients[0] * factor
        return indices, names, attrs, raw, weighted, gradients

    # This represents replacing f0 by factor*f0, including its residual and gradient.
    runtime.term_gradient_contributions = rescaled
    result = estimate(runtime)
    physical = np.asarray(result["weights"]) * [factor, 1.0]
    np.testing.assert_allclose(physical / physical.sum(), [0.75, 0.25], atol=1e-9)


@pytest.mark.parametrize("backend", ["pinocchio", "kots-rust"])
def test_robot_inverse_kinematics_recovers_known_weights(backend):
    from pathlib import Path

    urdf = Path(__file__).resolve().parents[1] / "examples/models/planar2.urdf"

    def make_runtime(weights):
        if backend == "pinocchio":
            pin = pytest.importorskip("pinocchio")
            if not hasattr(pin, "Model"):
                pytest.skip("Real Pinocchio is not installed.")
            from rei.backends.state.robotics.pinocchio import PinocchioStateBuilder
            model = pin.buildModelFromUrdf(str(urdf))
            builder = PinocchioStateBuilder(model, model.createData(), fields=("pos",))
        else:
            Kots = pytest.importorskip("robokots.kots").Kots
            from rei.backends.state.robotics.kots import KotsStateBuilder
            model = Kots.from_urdf_file(str(urdf), order=3)
            builder = KotsStateBuilder(model, fields=("pos",), dynamics_fields=None, kots_backend="rust")
        return compile_nls_problem_spec(
            {
                "opt_vals": {"joint_angles": {"init": [-0.2, 1.0]}},
                "terms": [
                    {"name": f"target_{i}", "state": "kinematics.link.ee.pos",
                     "frame": "world", "var": "joint_angles", "target": target,
                     "weight": float(weight)}
                    for i, (target, weight) in enumerate(zip(
                        [[1.3, 0.3, 0.0], [1.7, 0.7, 0.0]], weights,
                    ))
                ],
            }, build_state=builder.build_state,
        )

    demo = solve(make_runtime([0.75, 0.25]), options={"tol_grad": 1e-8, "tol_dx": 1e-8})
    assert demo.converged
    inference = make_runtime([1.0, 1.0])
    result = estimate(inference, p=demo.solution)
    np.testing.assert_allclose(result["weights"], [0.75, 0.25], atol=1e-7)
    assert result["stationarity"]["ikkt_residual_norm"] < 1e-8
    replay = solve(make_runtime(result["weights"]), options={"tol_grad": 1e-8, "tol_dx": 1e-8})
    assert replay.converged
    np.testing.assert_allclose(replay.solution, demo.solution, atol=1e-7)
