import json

import numpy as np
import pytest

from rei import solve, solve_lm_ls


class Problem:
    def __init__(self, x=(.1,), residual=lambda x: x*x - 1,
                 jacobian=lambda x: np.diag(2*x)):
        self.x = np.asarray(x, dtype=float)
        self.n_total = self.x.size
        self.residual, self.jacobian = residual, jacobian

    def get_point(self):
        return self.x.copy()

    def set_point(self, x):
        self.x = np.asarray(x, dtype=float).copy()

    def required_list(self, required=None):
        return [] if required is None else list(required)

    def eval(self, *, required=None):
        return self.residual(self.x)

    def linearize(self, *, required=None):
        return self.eval(), self.jacobian(self.x)


def test_backtracking_recovers_and_records_only_accepted_updates(tmp_path):
    problem = Problem()
    path = tmp_path / "history.jsonl"
    out = solve(problem, solver="lm-ls", options={
        "damping": .01, "verbose": False, "history_path": path})
    assert out.converged
    assert out.meta["gradient_converged"]
    assert np.max(np.abs(problem.jacobian(problem.x).T @ problem.residual(problem.x))) <= 1e-8
    trials = [e for e in out.line_search_history if e["event"] == "line_search_trial"]
    assert not trials[0]["accepted"]
    assert any(t["accepted"] and t["step_scale"] < 1 for t in trials)
    assert sum(t["accepted"] for t in trials) == out.iterations
    for t in trials:
        if t["accepted"]:
            assert t["base_objective"] - t["objective"] >= t["required_reduction"] > 0
    assert [json.loads(s) for s in path.read_text().splitlines()] == out.history
    trial_path = path.with_name("history.line_search.jsonl")
    assert [json.loads(s) for s in trial_path.read_text().splitlines()] == out.line_search_history


def test_failed_search_recomputes_damped_direction():
    out = solve_lm_ls(Problem(), max_iters=1, damping=.01, ls_min_step=1., verbose=False)
    retries = [e for e in out.line_search_history if e["event"] == "line_search_retry"]
    assert [e["damping"] for e in retries] == pytest.approx([.1, 1.])
    assert out.iterations == 1
    assert out.stats.objective < out.stats.initial_objective
    assert out.meta["reason"] == "max_iters"


def test_unrepresentable_step_is_not_convergence():
    problem = Problem(x=(1e16,), residual=lambda x: x - 1e16 + 1.,
                      jacobian=lambda x: np.eye(1))
    out = solve_lm_ls(problem, verbose=False)
    assert out.status == "stalled"
    assert not out.meta["gradient_converged"]
    assert out.iterations == 0
    np.testing.assert_array_equal(out.solution, [1e16])


def test_evaluation_failure_restores_last_accepted_point():
    def residual(x):
        if x[0] != .1:
            raise RuntimeError("evaluation failed")
        return x*x - 1
    problem = Problem(residual=residual)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        solve_lm_ls(problem, verbose=False)
    np.testing.assert_array_equal(problem.x, [.1])


@pytest.mark.parametrize("options", [{"ls_beta": 1.}, {"tol_grad": -1.},
    {"ls_max_retries": -1}, {"max_iters": 1.5}, {"damping": np.nan},
    {"tol_dx": 1e-12}])
def test_invalid_options_fail_before_mutation(options):
    problem = Problem()
    with pytest.raises(ValueError):
        solve(problem, solver="lm-ls", x0=[2.], options=options)
    np.testing.assert_array_equal(problem.x, [.1])


def test_gradient_default_and_zero_iteration_budget():
    problem = Problem(x=(1.,), residual=lambda x: x*5e-9,
                      jacobian=lambda x: np.eye(1))
    out = solve_lm_ls(problem, max_iters=0, history=False, verbose=False)
    assert out.converged and out.iterations == 0
    assert not out.history
    out = solve_lm_ls(Problem(), max_iters=0, verbose=False)
    assert out.status == "max_iters" and not out.meta["gradient_converged"]


def test_rosenbrock_reaches_gradient_target():
    problem = Problem(x=(-1.2, 1.),
        residual=lambda x: np.array([10*(x[1] - x[0]**2), 1-x[0]]),
        jacobian=lambda x: np.array([[-20*x[0], 10.], [-1., 0.]]))
    out = solve_lm_ls(problem, verbose=False)
    assert out.converged
    np.testing.assert_allclose(out.solution, [1., 1.], atol=1e-7)
    assert out.history[-1]["jt_r_inf_norm"] <= 1e-8
