import json

import numpy as np
import pytest

from functools import partial
from rei import solve as dispatch_solve, solve_gauss_newton

# These are regression tests for the preserved pre-LM algorithm.
solve = partial(dispatch_solve, solver="gauss_newton")


class ScalarProblem:
    n_total = 1

    def __init__(self, x=.1, residual=lambda x: x*x-1, jacobian=lambda x: 2*x):
        self.x = np.array([x], dtype=float)
        self.residual = residual
        self.jacobian = jacobian

    def get_point(self):
        return self.x.copy()

    def set_point(self, x):
        self.x = np.array(x, dtype=float)

    def required_list(self, required=None):
        return []

    def eval(self, *, required=None):
        return np.atleast_1d(self.residual(self.x[0]))

    def linearize(self, *, required=None):
        return self.eval(), np.asarray(self.jacobian(self.x[0])).reshape(-1, 1)


def test_extended_search_recovers_and_does_not_count_retries_as_updates(tmp_path):
    path = tmp_path / "history.jsonl"
    calls = []
    out = solve(ScalarProblem(), options={"ls_max_iters": 2, "history_path": path,
                                         "history_vectors": True},
                on_iter=lambda *args: calls.append(args))
    assert out.converged
    retries = [r for r in out.history if r["event"] == "iteration_retry"]
    assert retries[0]["reason"] == "expand_search"
    assert retries[0]["retry"] == 1
    assert retries[0]["ls_max_iters"] == 4
    assert retries[0]["damping"] == retries[0]["previous_damping"]
    assert retries[0]["variables"] == [.1]
    states = [r for r in out.history if r["event"] == "iteration_end"]
    assert states[0]["iteration"] == 1
    assert states[0]["line_search_trials"] == 5
    assert states[0]["retry"] == 1
    assert states[0]["step_scale"] == .25
    assert [r["iteration"] for r in states] == list(range(1, out.iterations + 1))
    assert len({args[0] for args in calls}) == len(calls)
    assert [json.loads(s) for s in path.read_text().splitlines()] == out.history
    rows = [json.loads(s) for s in path.with_name("history.line_search.jsonl").read_text().splitlines()]
    assert rows == out.line_search_history
    assert any(r["event"] == "line_search_retry" for r in rows)
    trials = [r for r in rows if r["event"] == "line_search_trial"]
    assert len({(r["iteration"], r["retry"], r["trial"]) for r in trials}) == len(trials)


def test_damping_retry_recomputes_direction_and_recovers():
    out = solve_gauss_newton(ScalarProblem(), max_iters=1, damping=.01, ls_min_step=1.,
                             history_vectors=True, verbose=False)
    retries = [r for r in out.history if r["event"] == "iteration_retry"]
    assert [r["reason"] for r in retries] == ["increase_damping", "increase_damping"]
    assert [r["damping"] for r in retries] == pytest.approx([.1, 1.])
    assert all(r["variables"] == [.1] for r in retries)
    trials = [r for r in out.line_search_history if r["event"] == "line_search_trial"]
    assert [r["accepted"] for r in trials] == [False, False, True]
    assert trials[0]["step_norm"] > trials[1]["step_norm"] > trials[2]["step_norm"]
    assert out.stats.objective < out.stats.initial_objective
    assert out.iterations == 1
    # The state record reports the damping to be used at that point, after
    # accepting and relaxing the preceding step.
    assert out.history[-2]["damping"] == .1


def test_default_retry_recovers_previously_stalled_exponential_problem():
    problem = ScalarProblem(-1.1, lambda x: np.exp(10*x)-1, lambda x: 10*np.exp(10*x))
    with np.errstate(over="ignore"):
        out = solve(problem, options={"verbose": False})
    assert out.converged
    assert out.solution == pytest.approx([0.], abs=1e-10)


@pytest.mark.parametrize("history", [True, False])
def test_retry_limit_does_not_mistake_tiny_damped_step_for_convergence(history):
    # The mathematical minimizer 1e16-1 is not representable in float64.
    problem = ScalarProblem(1e16, lambda x: x-1e16+1., lambda x: 1.)
    out = solve(problem, options={"damping": 0., "tol_grad": 1e-25,
                                  "ls_max_iters": 1, "ls_max_retries": 3,
                                  "verbose": False, "history": history})
    assert out.status == "stalled"
    assert out.iterations == 0
    assert out.solution == pytest.approx([1e16])
    if history:
        retries = [r for r in out.history if r["event"] == "iteration_retry"]
        assert len(retries) == 3
        assert [r["retry"] for r in retries] == [1, 2, 3]
        # The retry starts from a scale-aware numerical lower bound, rather
        # than an unrelated absolute bootstrap such as 1e-6.
        expected_floor = 100.0 * np.finfo(float).eps
        assert retries[0]["damping_min"] == pytest.approx(expected_floor)
        assert retries[0]["damping"] == pytest.approx(10.0 * expected_floor)
        assert out.history[-1]["reason"] == "line_search_retry_limit"
        assert out.history[-2]["line_search_trials"] == 4


def test_stationarity_is_checked_before_retry():
    problem = ScalarProblem(0, lambda x: [1., 1e-12*(x+1)], lambda x: [0., 1e-12])
    out = solve(problem, options={"damping": 0., "ls_max_iters": 1, "verbose": False})
    assert out.converged
    assert not any(r["event"] == "iteration_retry" for r in out.history)
    assert out.history[-1]["reason"] == "gradient_tolerance"


def test_damping_cap_stops_retries_without_changing_point():
    out = solve(ScalarProblem(), options={"damping": .01, "damping_max": .1,
                                         "ls_min_step": 1., "verbose": False})
    assert out.status == "stalled"
    assert out.solution == pytest.approx([.1])
    assert out.history[-1]["reason"] == "line_search_damping_limit"
    retries = [r for r in out.history if r["event"] == "iteration_retry"]
    assert len(retries) == 1 and retries[0]["damping"] == .1


@pytest.mark.parametrize("options", [
    {"ls_max_retries": -1}, {"ls_max_retries": 1.5}, {"ls_max_retries": np.inf},
    {"damping_increase": 1.}, {"damping_increase": np.nan},
    {"damping_max": 0.}, {"damping_max": np.inf},
    {"damping_decrease": 0.}, {"damping_decrease": 1.}, {"damping_decrease": np.nan},
    {"damping_min_factor": 0.}, {"damping_min_factor": np.nan},
    {"c_armijo": 0.}, {"c_armijo": 1.}, {"c_armijo": np.nan},
])
def test_retry_options_validated_before_evaluation(options):
    problem = ScalarProblem(residual=lambda x: pytest.fail("must validate first"))
    with pytest.raises(ValueError):
        solve(problem, options=options)


def test_damping_is_clamped_to_scale_aware_lower_bound():
    # max(diag(J.T @ J)) is 9 here, so even damping=0 must use λ_min.
    out = solve(ScalarProblem(1., lambda x: x, lambda x: 3.),
                options={"damping": 0., "damping_min_factor": 10.,
                         "max_iters": 1, "line_search": False, "verbose": False})
    initial = out.history[0]
    expected = 10.0 * np.finfo(float).eps * 9.0
    assert initial["damping_min"] == pytest.approx(expected)
    assert initial["damping"] == pytest.approx(expected)


@pytest.mark.parametrize("x0", [1e-8, 1e-12])
@pytest.mark.parametrize("history", [True, False])
def test_roundoff_cost_does_not_block_verified_gradient_progress(x0, history):
    problem = ScalarProblem(x0, lambda x: [np.sqrt(3.9), x], lambda x: [0., 1.])
    out = solve(problem, options={"damping": 0., "tol_dx": 1e-10, "tol_grad": 1e-14,
                                  "max_iters": 1, "history": history, "verbose": False})
    assert out.converged
    r, J = problem.linearize()
    assert np.max(np.abs(J.T @ r)) <= 1e-14
    assert out.iterations == 1
    if history:
        trial = next(r for r in out.line_search_history if r["event"] == "line_search_trial")
        assert trial["accepted"] and trial["reason"] == "gradient_progress"
        assert trial["step_scale"] == 1.
        assert trial["jt_r_inf_norm"] <= 1e-14


def test_success_relaxes_damping_and_continues_beyond_small_steps():
    problem = ScalarProblem(1., lambda x: x, lambda x: 1.)
    out = solve(problem, options={"damping": 1e6, "tol_dx": 1e-3, "verbose": False})
    assert out.converged
    states = [r for r in out.history if r["event"] == "iteration_end"]
    assert states[0]["step_norm"] < 1e-3
    assert states[0]["jt_r_inf_norm"] > 1e-10
    assert len(states) > 1
    assert states[1]["damping"] == states[1]["next_damping"]
    assert states[1]["damping"] < states[0]["damping"]
    assert out.history[-1]["jt_r_inf_norm"] <= 1e-10


def test_small_residual_alone_is_not_gradient_convergence():
    problem = ScalarProblem(0., lambda x: 1e12*x+1e-11, lambda x: 1e12)
    out = solve(problem, options={"tol_r": 1e-10, "tol_grad": 1e-10, "verbose": False})
    assert out.converged
    assert out.iterations > 0
    assert out.history[-1]["jt_r_inf_norm"] <= 1e-10


def test_tiny_roundoff_decrease_without_gradient_progress_is_rejected():
    # An evaluation perturbation below the numerical objective resolution must
    # not be taken as evidence of progress if the gradient stays unchanged.
    problem = ScalarProblem(0., lambda x: [np.nextafter(2., 0.) if x != 0 else 2., 1e-8],
                            lambda x: [0., 1.])
    out = solve(problem, options={"ls_max_retries": 0, "history_vectors": True,
                                  "verbose": False})
    assert out.status == "stalled"
    assert out.solution == pytest.approx([0.])
    trials = [r for r in out.line_search_history if r["event"] == "line_search_trial"]
    assert any(r["objective"] < r["base_objective"] for r in trials)
    assert all(not r["accepted"] for r in trials)
    assert all(r["reason"] == "roundoff_no_gradient_progress" for r in trials)


def test_armijo_rejects_insufficient_but_positive_decrease():
    out = solve(ScalarProblem(), options={"max_iters": 1, "c_armijo": .9, "verbose": False})
    trial = next(r for r in out.line_search_history
                 if r["event"] == "line_search_trial" and r["step_scale"] == .25)
    reduction = trial["base_objective"] - trial["objective"]
    assert 0 < reduction < trial["required_reduction"]
    assert not trial["accepted"]
    assert trial["reason"] == "insufficient_decrease"
