from __future__ import annotations

import json

import numpy as np
import pytest

from rei import solve, solve_levenberg_marquardt, format_solver_history


class Problem:
    def __init__(self, x=(.1,), residual=lambda x: x*x - 1,
                 jacobian=lambda x: np.diag(2*x)):
        self.x = np.asarray(x, dtype=float)
        self.n_total = self.x.size
        self.residual, self.jacobian = residual, jacobian
        self.linearizations = 0

    def get_point(self):
        return self.x.copy()

    def set_point(self, x):
        self.x = np.asarray(x, dtype=float).copy()

    def required_list(self, required=None):
        return [] if required is None else list(required)

    def eval(self, *, required=None):
        return self.residual(self.x)

    def linearize(self, *, required=None):
        self.linearizations += 1
        return self.eval(), self.jacobian(self.x)


def test_default_is_lm_and_legacy_is_still_selectable():
    assert solve(Problem(), options={"verbose": False}).meta["solver"] == "levenberg_marquardt"
    legacy = solve(Problem(), solver="gauss_newton", options={"verbose": False})
    assert legacy.meta["solver"] == "gauss_newton"
    assert legacy.converged and legacy.line_search_history
    assert not legacy.trial_history


def test_rejection_keeps_state_and_linearization_and_uses_nielsen_update():
    problem = Problem()
    out = solve(problem, options={"max_iters": 4, "damping": .01,
                                 "history_vectors": True, "verbose": False})
    trials = out.trial_history
    assert [row["damping"] for row in trials[:3]] == pytest.approx([.01, .02, .08])
    assert not trials[0]["accepted"] and not trials[1]["accepted"]
    states = [row for row in out.history if row["event"] == "iteration_end"]
    for state, trial in zip(states, trials):
        if not trial["accepted"]:
            assert state["variables"] == [.1]
            assert state["step_norm"] == 0
            assert state["delta_objective"] == 0
        else:
            rho = trial["gain_ratio"]
            assert trial["next_damping"] == pytest.approx(
                trial["damping"] * max(1/3, 1-(2*rho-1)**3))
        assert trial["gain_ratio"] == pytest.approx(trial["actual_reduction"] / trial["predicted_reduction"])
    assert problem.linearizations == 1 + sum(t["accepted"] for t in trials)
    np.testing.assert_array_equal(problem.x, out.solution)


def test_rosenbrock_matches_independent_textbook_reference():
    def residual(x):
        return np.array([10*(x[1]-x[0]**2), 1-x[0]])

    def jacobian(x):
        return np.array([[-20*x[0], 10], [-1, 0]])

    x = np.array([-1.2, 1.])
    r, J = residual(x), jacobian(x)
    mu, nu = 1e-3 * np.max(np.diag(J.T @ J)), 2.
    expected = []
    # Independent normal-equation implementation, before any stopping threshold.
    for _ in range(12):
        g = J.T @ r
        h = np.linalg.solve(J.T @ J + mu*np.eye(2), -g)
        new_r = residual(x+h)
        rho = (r@r - new_r@new_r) / (h @ (mu*h - g))
        accepted = rho > 0
        old_mu = mu
        if accepted:
            x, r = x+h, new_r
            J = jacobian(x)
            mu, nu = mu*max(1/3, 1-(2*rho-1)**3), 2.
        else:
            mu, nu = mu*nu, 2*nu
        expected.append((x.copy(), old_mu, mu, rho, accepted))
    out = solve(Problem((-1.2, 1.), residual, jacobian), options={
        "max_iters": 12, "history_vectors": True, "verbose": False})
    states = [row for row in out.history if row["event"] == "iteration_end"]
    for row, (x, mu, next_mu, rho, accepted) in zip(states, expected, strict=True):
        np.testing.assert_allclose(row["variables"], x, atol=1e-12)
        assert row["damping"] == pytest.approx(mu)
        assert row["next_damping"] == pytest.approx(next_mu)
        assert row["gain_ratio"] == pytest.approx(rho)
        assert row["accepted"] == accepted
    converged = solve(Problem((-1.2, 1.), residual, jacobian), options={"verbose": False})
    assert converged.converged
    np.testing.assert_allclose(converged.solution, [1, 1], atol=1e-8)


def test_linear_solution_rank_deficiency_and_last_iteration_check():
    out = solve(Problem([0., 0.], lambda x: np.array([x.sum()-1]),
                        lambda x: np.ones((1, 2))), options={"verbose": False})
    assert out.converged
    np.testing.assert_allclose(out.solution, [.5, .5], atol=1e-8)
    last = solve(Problem([0.], lambda x: x-1, lambda x: np.eye(1)),
                 options={"max_iters": 1, "damping": 1e-12, "verbose": False})
    assert last.converged and last.iterations == 1


def test_step_stop_is_distinguished_from_stationarity():
    out = solve(Problem(), options={"damping": 1e6, "tol_dx": 1e-3, "verbose": False})
    assert out.converged and out.meta["reason"] == "step_tolerance"
    assert out.meta["gradient_converged"] is False
    assert out.solution == pytest.approx([.1])
    strict = solve(Problem(), options={"damping": 1e6, "tol_dx": 0, "max_iters": 1, "verbose": False})
    assert strict.status == "max_iters"
    assert strict.trial_history[0]["accepted"]


def test_roundoff_gradient_progress_is_not_a_baseline_acceptance_rule():
    problem = Problem([0.], lambda x: np.array([1., x[0]-1e-9]),
                      lambda x: np.array([[0.], [1.]]))
    out = solve(problem, options={"max_iters": 1, "tol_dx": 0,
                                 "tol_grad": 1e-12, "verbose": False})
    trial = out.trial_history[0]
    assert trial["predicted_reduction"] > 0
    assert trial["actual_reduction"] == 0
    assert not trial["accepted"]
    assert out.status == "max_iters"
    np.testing.assert_array_equal(out.solution, [0.])


def test_stationary_initial_point_and_no_variables():
    for problem in (Problem([1.]), Problem([], lambda x: np.array([1.]),
                                          lambda x: np.zeros((1, 0)))):
        out = solve(problem, options={"verbose": False})
        assert out.converged and out.iterations == 0
        assert out.meta["gradient_converged"]
        assert not out.trial_history


def test_trial_only_stream_and_file_collision(tmp_path):
    path = tmp_path / "trials.jsonl"
    out = solve(Problem(), options={"max_iters": 1, "history": False,
                                  "trial_history_path": path, "verbose": False})
    assert not out.history and not out.trial_history
    assert json.loads(path.read_text())["event"] == "lm_trial"
    with pytest.raises(FileExistsError):
        solve(Problem(), options={"trial_history_path": path})
    unused = tmp_path / "unused.jsonl"
    with pytest.raises(ValueError, match="different files"):
        solve(Problem(), options={"history_path": unused, "trial_history_path": unused})
    assert not unused.exists()


@pytest.mark.parametrize("max_iters", [0, 1, 30])
def test_history_roundtrip_and_initial_delta(tmp_path, max_iters):
    path = tmp_path / "history.jsonl"
    out = solve(Problem(), options={"max_iters": max_iters, "history_path": path,
                                  "history_vectors": True, "verbose": False})
    assert out.history[0]["delta_objective"] == 0
    assert [json.loads(line) for line in path.read_text().splitlines()] == out.history
    assert [json.loads(line) for line in path.with_name("history.lm_trials.jsonl").read_text().splitlines()] == out.trial_history
    assert not out.line_search_history
    assert out.history[-1]["variables"] == out.solution.tolist()
    assert all(row["solver"] == "levenberg_marquardt" for row in out.history + out.trial_history)
    assert "damping=" in format_solver_history(out.history)
    if max_iters:
        assert "rho=" in format_solver_history(out.history)


def test_nonfinite_trial_is_rejected_and_exception_restores_point():
    def residual(x):
        return np.full(1, np.inf) if x[0] > 1 else x*x-1

    problem = Problem(residual=residual)
    out = solve(problem, options={"max_iters": 1, "verbose": False})
    assert out.trial_history[0]["reason"] == "nonfinite_objective"
    np.testing.assert_array_equal(problem.x, [.1])

    def failing(x):
        if x[0] > 1:
            raise RuntimeError("model failure")
        return x*x-1

    problem = Problem(residual=failing)
    with pytest.raises(RuntimeError, match="model failure"):
        solve(problem, options={"verbose": False})
    np.testing.assert_array_equal(problem.x, [.1])


@pytest.mark.parametrize("options", [{"tau": 0}, {"damping": 0}, {"damping": np.inf},
                                    {"tol_grad": -1}, {"tol_dx": np.nan}, {"max_iters": 1.5}])
def test_validation_precedes_mutation(options):
    problem = Problem()
    with pytest.raises(ValueError):
        solve(problem, x0=[2.], options=options)
    np.testing.assert_array_equal(problem.x, [.1])


@pytest.mark.parametrize("options", [{"line_search": False}, {"ls_max_retries": 2}, {"c_armijo": .01}])
def test_baseline_rejects_legacy_options(options):
    with pytest.raises(ValueError, match="unsupported option"):
        solve(Problem(), options=options)


def test_direct_export_and_callback_and_history_disabled():
    calls = []
    out = solve_levenberg_marquardt(Problem(), history=False, verbose=False,
                                   on_iter=lambda k, r, h, g: calls.append((k, g)))
    assert out.converged and calls
    assert not out.history and not out.trial_history
