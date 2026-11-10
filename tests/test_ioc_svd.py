"""Known-answer SVD IOC, invalid directions and reduced-coordinate integration."""
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from rei import compile_nls_problem, compile_nls_problem_spec, solve
from rei.equations import SVDWeightError, StationarityTermContribution, estimate_stationarity_weights
from rei.equations.report import build_ioc_log_sections, format_ioc_report
from rei.optimize.reductions import build_nullspace_equality_reduction
from rei.optimize_backends.trajectory_ioc import estimate_ioc_weights


class MatrixSource:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float)
        self.n_total = self.matrix.shape[0]

    def required_list(self, required=None):
        return []

    def term_contributions(self, *, required=None):
        return [StationarityTermContribution(
            i + 5, f"objective_{i}", {}, column,
            residual_norm=1., weighted_residual_norm=1., reference_weight=float(i + 1),
        ) for i, column in enumerate(self.matrix.T)]


def estimate(matrix, **kwargs):
    return estimate_stationarity_weights(MatrixSource(matrix), weight_method="svd", **kwargs)


@pytest.mark.parametrize("rows", [1, 3, 200])
def test_known_weights_include_wide_and_tall_matrices(rows):
    matrix = np.outer(np.arange(1., rows + 1), [.5, -1.5])
    result = estimate(matrix)
    np.testing.assert_allclose(result["weights"], [.75, .25], atol=1e-14)
    np.testing.assert_allclose(result["scaled_weights"], result["weights"])
    np.testing.assert_allclose(result["stationarity"]["ikkt_residual"], matrix @ result["weights"])
    assert result["stationarity_scaling"]["mode"] == "none"
    assert result["stationarity_scaling"]["column_scale"] == [1., 1.]
    assert result["weight_method"] == "svd" and result["simplex"] is None
    assert result["svd"]["nullity"] == 1
    assert result["svd"]["unique_direction"]
    assert result["svd"]["implicit_zero_singular_values"] == (1 if rows == 1 else 0)
    assert result["weight_solve_seconds"] >= 0
    assert result["identifiability"]["unique_stationary_weights"]
    assert [term["term_index"] for term in result["terms"]] == [5, 6]
    json.dumps(result, allow_nan=False)


def test_noisy_fit_records_spectrum_and_actual_nonzero_residual():
    matrix = np.array([[1., -2.], [.3, -.4]])
    result = estimate(matrix, stationarity_scaling="none")
    w = np.asarray(result["weights"])
    _, s, vh = np.linalg.svd(matrix, full_matrices=False)
    direction = vh[-1] * np.sign(vh[-1].sum())
    np.testing.assert_allclose(w, direction / direction.sum())
    np.testing.assert_allclose(result["svd"]["singular_values"], s)
    np.testing.assert_allclose(result["stationarity"]["ikkt_residual_norm"], np.linalg.norm(matrix @ w))
    assert result["stationarity"]["ikkt_residual_norm"] > .01
    assert not result["identifiability"]["unique_stationary_weights"]


def test_mixed_signs_rejected_without_projection():
    with pytest.raises(SVDWeightError, match="mixed signs") as error:
        estimate([[1., 2.]])
    assert error.value.diagnostics["status"] == "mixed_signs"
    assert error.value.diagnostics["active_term_indices"] == [5, 6]
    json.dumps(error.value.diagnostics, allow_nan=False)


@pytest.mark.parametrize("matrix,multiplicity", [
    ([[1., 1., -3.]], 2),
    (np.eye(2), 2),
    ([[1., -1., 1., -1.]], 3),
])
def test_repeated_minimum_rejected_with_diagnostics(matrix, multiplicity):
    with pytest.raises(SVDWeightError, match="multiplicity") as error:
        estimate(matrix)
    info = error.value.diagnostics
    assert info["status"] == "non_unique_direction"
    assert info["smallest_multiplicity"] == multiplicity
    assert not info["unique_direction"]


def test_relative_tolerance_detects_near_repeated_minimum():
    matrix = np.array([[1., 0.], [0., 1. + 1e-8]])
    assert estimate(matrix)["weights"] == [1., 0.]
    with pytest.raises(SVDWeightError, match="multiplicity"):
        estimate(matrix, svd_rtol=1e-6)


def test_only_roundoff_negative_weights_are_clipped():
    result = estimate([[1., 1e-12]], active_mode="residual")
    np.testing.assert_array_equal(result["weights"], [0., 1.])
    with pytest.raises(SVDWeightError, match="mixed signs"):
        estimate([[1., 1e-12]], active_mode="residual", svd_nonnegative_tol=0.)


def test_inactive_terms_keep_original_positions_and_no_false_identifiability():
    result = estimate([[.5, 0., -1.5]])
    np.testing.assert_allclose(result["weights"], [.75, 0., .25])
    assert result["stationarity"]["active_indices"] == [0, 2]
    assert result["identifiability"]["excluded_term_indices"] == [6]
    assert not result["identifiability"]["unique_stationary_weights"]


def test_no_active_terms_are_not_reported_as_success():
    with pytest.raises(SVDWeightError, match="active objective") as error:
        estimate([[0., 0.]])
    assert error.value.diagnostics["status"] == "no_active_terms"


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_gradients_rejected_including_inactive_columns(value):
    with pytest.raises(ValueError, match="finite"):
        estimate([[.5, -1.5, value]])


@pytest.mark.parametrize("option", ["svd_rtol", "svd_nonnegative_tol"])
@pytest.mark.parametrize("value", [-1., 1., np.nan, np.inf])
def test_invalid_tolerances(option, value):
    with pytest.raises(ValueError, match=option):
        estimate([[.5, -1.5]], **{option: value})


def test_scaled_svd_and_kkt_mode_are_rejected():
    with pytest.raises(ValueError, match="stationarity_scaling='none'"):
        estimate([[.5, -1.5]], stationarity_scaling="gradient_norm")
    with pytest.raises(ValueError, match="incompatible"):
        estimate_ioc_weights(None, weight_method="svd", constraint_mode="kkt")
    with pytest.raises(ValueError, match="weight_method"):
        estimate_stationarity_weights(MatrixSource([[1.]]), weight_method="unknown")


def test_forward_reconstruction_and_adapter_preserve_weight_scale():
    def runtime(weights):
        return compile_nls_problem_spec({
            "opt_vals": {"x": {"init": [.5]}},
            "terms": [{"name": f"target_{i}", "var": "x", "target": [target], "weight": weight}
                      for i, (target, weight) in enumerate(zip([0., 2.], weights))],
        }, build_state=lambda *args, **kwargs: {})
    result = estimate_ioc_weights(SimpleNamespace(runtime=runtime([1., 1.])), weight_method="svd")
    np.testing.assert_allclose(result["weights"], [.75, .25])
    reconstructed = solve(runtime(result["weights"]), options={"verbose": False})
    np.testing.assert_allclose(reconstructed.solution, [.5], atol=1e-8)


def test_svd_uses_existing_nullspace_reduced_gradients():
    # A full-space gradient in y is balanced by the equality multiplier.
    # Only x stationarity identifies the weights after eliminating y=0.
    def target(value):
        return {"type": "sub", "a": {"type": "get_var", "var": "x"},
                "b": {"type": "const", "var": "x", "value": value}}
    full = compile_nls_problem({
        "variables": [{"name": "x", "dim": 2, "init": [.5, 0.]}],
        "terms": [
            {"expr": target([0., 0.]), "cost": {"type": "l2"}},
            {"expr": target([2., 2.]), "cost": {"type": "l2"}},
            {"expr": {"type": "component", "segment_dim": 2, "index": 1,
                      "base": {"type": "get_var", "var": "x"}},
             "cost": {"type": "l2"}, "constraint": {"kind": "eq"}},
        ],
    }, build_state=lambda *args, **kwargs: {})
    reduction = build_nullspace_equality_reduction(full, eq_term_indices=[2])
    result = estimate_ioc_weights(SimpleNamespace(runtime=reduction.runtime), weight_method="svd")
    np.testing.assert_allclose(result["weights"], [.75, .25], atol=1e-14)
    assert result["svd"]["matrix_shape"] == [1, 2]
    assert result["stationarity"]["ikkt_residual_norm"] < 1e-12


def test_svd_preserves_report_errors_and_adds_spectrum_and_timing():
    result = estimate([[.5, -1.5]])
    stationarity = result["stationarity"]
    arguments = dict(
        active_mode="gradient", active_idx=stationarity["active_indices"],
        active_grad_idx=stationarity["active_gradient_indices"],
        active_res_idx=stationarity["active_residual_indices"],
        term_indices=stationarity["term_indices"], w_hat=result["weights"], w_true=[.75, .25],
        ikkt_residual=stationarity["ikkt_residual_norm"], svd_result=result["svd"],
        weight_solve_seconds=result["weight_solve_seconds"],
    )
    report = format_ioc_report(**arguments)
    assert "L1 error=" in report and "svd: status=estimated" in report
    assert "singular_values=" in report and "weight_solve_seconds=" in report
    sections = dict(build_ioc_log_sections(**arguments))
    assert "ioc.result" in sections and "svd.result" in sections
    assert sections["svd.result"]["weight_solve_seconds"] == result["weight_solve_seconds"]
    json.dumps(sections, allow_nan=False)


def test_default_simplex_keeps_its_scaling_and_solver():
    result = estimate_stationarity_weights(MatrixSource([[.5, -1.5]]))
    assert result["weight_method"] == "simplex" and result["svd"] is None
    assert result["stationarity_scaling"]["mode"] == "gradient_norm"
    np.testing.assert_allclose(result["scaled_weights"], [.5, .5])
    assert result["simplex"]["method"] == "qr_nullspace"


def test_bspline_doc_to_svd_ioc_keeps_boundary_reduction_and_term_order():
    # Both target trajectories lie in the cubic spline space and share the
    # boundary values. Their known weighted mean is the exact DOC solution.
    samples = np.linspace(0., 1., 9)
    terms = []
    for index, (sign, weight) in enumerate(zip([1., -1.], [.75, .25])):
        parts = [{"type": "sub", "a": {"type": "get_traj_var", "var": "p", "k": k},
                  "b": {"type": "const", "var": "p", "dim": 1, "value": [u + sign * u * (1-u)]}}
                 for k, u in enumerate(samples)]
        terms.append({"expr": {"type": "vstack", "name": f"target_{index}", "parts": parts},
                      "cost": {"type": "scalar_weight", "w": weight}})
    for k, value in [(0, 0.), (8, 1.)]:
        terms.append({"expr": {"type": "sub", "a": {"type": "get_traj_var", "var": "p", "k": k},
                               "b": {"type": "const", "var": "p", "dim": 1, "value": [value]}},
                      "cost": {"type": "l2"}, "constraint": {"kind": "eq"}})
    dsl = {"time": {"N": 8, "dt": .125},
           "trajectory": {"type": "bspline", "var": "p", "degree": 3,
                          "num_ctrl_points": 6, "q_dim": 1},
           "variables": [{"name": "p", "dim": 6, "init": [0.] * 6}], "terms": terms}

    def compile_reduced(problem):
        runtime = compile_nls_problem(problem, build_state=lambda *args, **kwargs: {})
        return build_nullspace_equality_reduction(runtime, eq_term_indices=[2, 3])

    doc = compile_reduced(dsl)
    outcome = solve(doc.runtime, solver="gauss_newton",
                    options={"verbose": False, "tol_grad": 1e-12, "tol_dx": 1e-14})
    assert outcome.converged
    full_demo = doc.lift(outcome.solution)
    inverse_dsl = deepcopy(dsl)
    for term in inverse_dsl["terms"][:2]:
        term["cost"]["w"] = 1.
    inverse = compile_reduced(inverse_dsl)
    result = estimate_ioc_weights(SimpleNamespace(runtime=inverse.runtime),
                                  p=inverse.project(full_demo), weight_method="svd")
    np.testing.assert_allclose(result["weights"], [.75, .25], atol=1e-9)
    assert [term["name"] for term in result["terms"]] == ["target_0", "target_1"]
    assert result["svd"]["matrix_shape"] == [4, 2]
    assert result["stationarity"]["ikkt_residual_norm"] < 1e-10
