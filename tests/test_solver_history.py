from __future__ import annotations

import json

import numpy as np
import pytest

from functools import partial
from rei import format_solver_history, solve as dispatch_solve
from rei.optimize.textlog import format_solver_text_log

# Preserve the original Gauss-Newton history contract explicitly.
solve = partial(dispatch_solve, solver="gauss_newton")


@pytest.mark.parametrize("solver, trial_kind", [("gauss_newton", "line_search"),
                                               ("levenberg_marquardt", "lm")])
@pytest.mark.parametrize("history", [True, False])
def test_verbose_repeats_header_every_twenty_iterations(capsys, solver, trial_kind, history):
    from rei.optimize.history import SolverHistoryRecorder

    recorder = SolverHistoryRecorder(enabled=history, verbose=True,
                                     solver=solver, trial_kind=trial_kind)
    recorder.emit("initial", 0, objective=1.)
    for k in range(1, 42):
        recorder.emit(trial_kind + "_trial", k, trial=1, objective=1.)
        if k % 20 == 0:
            # Multiple events within a boundary iteration must not repeat it.
            for retry in (1, 2):
                recorder.emit("iteration_retry", k, retry=retry, damping=.01, ls_max_iters=12)
        recorder.emit("iteration_end", k, objective=1.)
        if k % 20 == 0:
            recorder.emit("final", k, objective=1.)
    lines = capsys.readouterr().out.splitlines()
    header_indices = [i for i, line in enumerate(lines) if line.split()[0] == "iter"]
    assert len(header_indices) == 3
    assert [int(lines[i+1].split()[0]) for i in header_indices] == [0, 20, 40]
    assert all(lines[i] == lines[0] for i in header_indices)
    assert ("λ" in lines[0]) == (solver == "levenberg_marquardt")
    if history:
        assert all(row["event"] != "header" for row in recorder.events)
    else:
        assert not recorder.events


class ScalarProblem:
    n_total = 1

    def __init__(self, x=0.1, residual=lambda x: x*x - 1, jacobian=lambda x: 2*x):
        self.x = np.array([x], dtype=float)
        self.residual = residual
        self.jacobian = jacobian

    def get_point(self):
        return self.x.copy()

    def set_point(self, x):
        self.x = np.asarray(x, dtype=float).copy()

    def required_list(self, required=None):
        return []

    def eval(self, *, required=None):
        return np.atleast_1d(self.residual(self.x[0]))

    def linearize(self, *, required=None):
        return self.eval(), np.asarray(self.jacobian(self.x[0])).reshape(-1, 1)


def test_history_backtracking_and_state_consistency(tmp_path):
    path = tmp_path / "history.jsonl"
    out = solve(ScalarProblem(), options={"history_path": path, "history_vectors": True})
    assert out.converged
    assert [json.loads(line) for line in path.read_text().splitlines()] == out.history
    assert [json.loads(line) for line in (tmp_path / "history.line_search.jsonl").read_text().splitlines()] == out.line_search_history
    assert not any(row["event"].startswith("line_search_") for row in out.history)
    trials = [row for row in out.line_search_history if row["event"] == "line_search_trial" and row["iteration"] == 1]
    assert [row["step_scale"] for row in trials] == [1, .5, .25]
    assert [row["accepted"] for row in trials] == [False, False, True]
    assert [row["trial"] for row in trials] == [1, 2, 3]
    states = [row for row in out.history if row["event"] in {"initial", "iteration_end"}]
    assert [row["iteration"] for row in states] == list(range(out.iterations + 1))
    assert states[0]["step_norm"] is None
    assert states[0]["delta_objective"] == 0.0
    assert states[1]["line_search_trials"] == 3
    assert states[1]["step_scale"] == .25
    assert states[1]["line_search_status"] == "accepted"
    for row in states:
        x = row["variables"][0]
        assert row["objective"] == pytest.approx((x*x - 1)**2)
        assert row["jt_r"] == pytest.approx([2*x*(x*x-1)])
        assert row["jt_r_inf_norm"] == pytest.approx(abs(row["jt_r"][0]))
    for before, after in zip(states, states[1:]):
        assert after["delta_objective"] == pytest.approx(after["objective"] - before["objective"])
        assert after["step_norm"] == pytest.approx(abs(after["variables"][0] - before["variables"][0]))
    final = out.history[-1]
    assert final["event"] == "final"
    assert final["status"] == out.status
    assert final["objective"] == out.stats.objective
    assert final["residual_norm"] == out.stats.residual_norm
    assert final["step_norm"] == out.stats.step_norm
    assert final["variables"] == out.solution.tolist()
    elapsed = [row["elapsed_seconds"] for row in out.history]
    assert elapsed == sorted(elapsed)


@pytest.mark.parametrize("options,reason,count", [
    ({"ls_max_iters": 1}, "max_trials", 1),
    ({"ls_min_step": .75}, "step_too_small", 1),
    ({"ls_min_step": 2}, "step_too_small", 0),
])
def test_failed_search_retains_point(options, reason, count):
    out = solve(ScalarProblem(), options={**options, "ls_max_retries": 0})
    assert out.status == "stalled"
    assert out.solution == pytest.approx([.1])
    assert len([row for row in out.line_search_history if row["event"] == "line_search_trial"]) == count
    assert out.history[-2]["event"] == "iteration_failed"
    assert out.history[-2]["iteration"] == 1
    assert out.history[-2]["line_search_status"] == reason
    assert out.history[-2]["line_search_trials"] == count
    assert out.history[-2]["step_scale"] is None
    assert out.history[-1]["reason"] == "line_search_" + reason
    assert not any(row["event"] == "iteration_end" for row in out.history)
    assert out.history[-1]["objective"] == out.history[0]["objective"]
    assert out.history[-2]["delta_objective"] == 0.0


@pytest.mark.parametrize("tol_grad,updates", [(1e-23, 0), (1e-24, 0), (1e-25, 1)])
@pytest.mark.parametrize("history", [True, False])
def test_gradient_checked_even_when_objective_change_is_unresolvable(tol_grad, updates, history):
    points = []

    class RoundedObjective(ScalarProblem):
        def linearize(self, *, required=None):
            points.append(self.x.copy())
            return super().linearize(required=required)

    # A full GN direction of -1, but its cost reduction is lost to rounding.
    problem = RoundedObjective(x=0, residual=lambda x: [1., 1e-12*(x+1)],
                               jacobian=lambda x: [0., 1e-12])
    out = solve(problem, options={"damping": 0., "tol_grad": tol_grad,
                                  "ls_max_iters": 2, "ls_max_retries": 0,
                                  "history": history, "verbose": False})
    assert out.converged
    assert out.iterations == updates
    assert out.solution == pytest.approx([0.] if updates == 0 else [-1.], abs=1e-3)
    assert out.stats.step_norm == 0.
    assert len(points) == updates + 1
    assert points[-1] == pytest.approx(out.solution)
    if history:
        final = out.history[-1]
        assert final["jt_r_inf_norm"] <= tol_grad
        assert final["status"] == out.status
        assert final["reason"] == "gradient_tolerance"


@pytest.mark.parametrize("max_iters", [0, 1])
def test_iteration_limit_and_no_search(max_iters):
    out = solve(ScalarProblem(), options={"max_iters": max_iters, "line_search": False})
    assert out.status == "max_iters"
    assert out.history[0]["event"] == "initial"
    assert out.history[-1]["iteration"] == max_iters
    assert not any(row["event"].startswith("line_search") for row in out.history)
    assert all(row.get("step_norm") is None or np.isfinite(row["step_norm"]) for row in out.history)
    assert "variables" not in out.history[0]


def test_initial_convergence():
    out = solve(ScalarProblem(x=1))
    assert [row["event"] for row in out.history] == ["initial", "final"]
    assert out.history[-1]["reason"] == "residual_tolerance"
    assert out.history[0]["delta_objective"] == 0.0
    assert out.history[-1]["delta_objective"] == 0.0


@pytest.mark.parametrize("max_iters", [0, 1])
def test_initial_delta_objective_in_stream_and_display(tmp_path, max_iters):
    path = tmp_path / "history.jsonl"
    solve(ScalarProblem(), options={"history": False, "history_path": path, "max_iters": max_iters})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["delta_objective"] == 0.0
    text = format_solver_history(rows)
    assert "Δobj" in text.splitlines()[0]
    assert text.splitlines()[1].split()[3] == "0.00e+00"


def test_small_step_does_not_stop_before_gradient_converges():
    out = solve(ScalarProblem(), options={"tol_dx": 10, "history_vectors": True})
    assert out.status == "converged"
    assert out.history[-1]["jt_r_inf_norm"] <= 1e-10
    assert out.history[-2]["event"] == "iteration_end"
    assert out.history[-1]["variables"] == out.solution.tolist()


def test_nonfinite_trials_are_strict_json(tmp_path):
    path = tmp_path / "history.jsonl"
    problem = ScalarProblem(x=-1.1, residual=lambda x: np.exp(10*x)-1, jacobian=lambda x: 10*np.exp(10*x))
    with np.errstate(over="ignore"):
        out = solve(problem, options={"history_path": path})
    trials = [row for row in out.line_search_history if row.get("reason") == "non_finite"]
    assert trials
    assert trials[0]["objective"] is None
    assert trials[0]["accepted"] is False
    assert "Infinity" not in path.read_text()
    assert "NaN" not in path.read_text()
    assert "Infinity" not in (tmp_path / "history.line_search.jsonl").read_text()


def test_stream_only_and_disabled_history(tmp_path):
    out = solve(ScalarProblem(), options={"history": False})
    assert out.history == []
    path = tmp_path / "history.jsonl"
    streamed = solve(ScalarProblem(), options={"history": False, "history_path": path})
    assert streamed.history == []
    assert streamed.line_search_history == []
    assert json.loads(path.read_text().splitlines()[-1])["status"] == streamed.status
    assert streamed.solution == pytest.approx(out.solution)


def test_never_overwrite_existing_history(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text("previous run\n")
    problem = ScalarProblem()
    with pytest.raises(FileExistsError):
        solve(problem, options={"history_path": path})
    assert path.read_text() == "previous run\n"
    assert problem.get_point() == pytest.approx([.1])


def test_trial_error_preserves_partial_history_and_point(tmp_path):
    def residual(x):
        if x > 1:
            raise ValueError("outside domain")
        return x*x - 1
    problem = ScalarProblem(residual=residual)
    path = tmp_path / "history.jsonl"
    with pytest.raises(ValueError, match="outside domain"):
        solve(problem, options={"history_path": path, "history_vectors": True})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["event"] == "initial"
    assert rows[-1]["line_search_status"] == "evaluation_error"
    assert rows[-1]["event"] == "iteration_failed"
    assert rows[-1]["variables"] == rows[0]["variables"]
    search = [json.loads(line) for line in (tmp_path / "history.line_search.jsonl").read_text().splitlines()]
    assert search[-1]["event"] == "line_search_end"
    assert search[-1]["reason"] == "evaluation_error"
    assert problem.get_point() == pytest.approx([.1])


def test_human_readable_history_and_log():
    out = solve(ScalarProblem())
    text = format_solver_history(out.history)
    assert "trial 1" not in text and "accepted" in text
    details = format_solver_history(out.line_search_history)
    assert "trial 1" in details and "insufficient_decrease" in details
    assert "|Jᵀr|inf" in text
    assert "trial 1" not in format_solver_history(out.history, include_line_search=False)
    log = format_solver_text_log(title="test", solver="gauss_newton", outcome=out)
    assert "[solve.history]" in log and "trial 1" not in log


def test_explicit_line_search_path_and_collision(tmp_path):
    path = tmp_path / "history.jsonl"
    search = tmp_path / "line_search.jsonl"
    out = solve(ScalarProblem(), options={"history_path": path, "line_search_history_path": search})
    assert search.exists()
    assert not (tmp_path / "history.line_search.jsonl").exists()
    assert [json.loads(line) for line in search.read_text().splitlines()] == out.line_search_history
    unused = tmp_path / "unused.jsonl"
    with pytest.raises(FileExistsError):
        solve(ScalarProblem(), options={"history_path": unused, "line_search_history_path": search})
    assert not unused.exists()
    with pytest.raises(ValueError, match="different files"):
        solve(ScalarProblem(), options={"history_path": unused, "line_search_history_path": unused})
    assert not unused.exists()


def test_line_search_only_stream(tmp_path):
    path = tmp_path / "search.jsonl"
    out = solve(ScalarProblem(), options={"history": False, "line_search_history_path": path})
    assert out.history == out.line_search_history == []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows and all(row["event"].startswith("line_search_") for row in rows)


def test_disabled_search_has_empty_detail_file(tmp_path):
    path = tmp_path / "history.jsonl"
    out = solve(ScalarProblem(), options={"max_iters": 1, "line_search": False, "history_path": path})
    assert out.line_search_history == []
    assert (tmp_path / "history.line_search.jsonl").read_text() == ""
    assert out.history[-2]["line_search_status"] == "disabled"
    assert out.history[-2]["line_search_trials"] == 0


def test_legacy_callback_contract_is_unchanged():
    events = []
    solve(ScalarProblem(), options={"max_iters": 1}, on_iter=lambda k, r, dx: events.append((k, r, dx)))
    assert len(events) == 1
    assert events[0][0] == 0
    assert events[0][1] == pytest.approx(.99)


def test_gauss_newton_prints_live_history_without_callback(capsys):
    class ObservedProblem(ScalarProblem):
        def eval(self, *, required=None):
            if self.x[0] > 2:
                # The initial row must already be printed before the first trial.
                assert "initial" in capsys.readouterr().out
            return super().eval(required=required)

    out = solve(ObservedProblem(), options={"ls_max_iters": 1, "ls_max_retries": 0})
    text = capsys.readouterr().out
    assert "iteration_failed" in text
    assert "final" in text
    assert "trial 1" not in text
    assert out.status == "stalled"


def test_live_output_matches_saved_history(capsys):
    out = solve(ScalarProblem())
    assert capsys.readouterr().out.rstrip() == format_solver_history(out.history).rstrip()


def test_live_output_uses_three_significant_digits_without_rounding_storage(capsys, tmp_path):
    path = tmp_path / "history.jsonl"
    out = solve(ScalarProblem(x=.123456), options={"max_iters": 0, "history_path": path})
    initial = capsys.readouterr().out.splitlines()[1].split()
    assert initial[2] == f"{out.history[0]['objective']:.2e}"
    assert initial[3] == "0.00e+00"
    assert initial[4] == f"{out.history[0]['jt_r_inf_norm']:.2e}"
    saved = json.loads(path.read_text().splitlines()[0])
    assert saved["objective"] == out.history[0]["objective"]
    assert saved["objective"] != float(initial[2])


def test_verbose_independent_of_history_storage(capsys):
    out = solve(ScalarProblem(), options={"history": False})
    assert out.history == []
    assert "iteration_end" in capsys.readouterr().out
    out = solve(ScalarProblem(), options={"verbose": False})
    assert out.history
    assert capsys.readouterr().out == ""


def test_compact_columns_keep_large_and_negative_values_separate():
    text = format_solver_history([{
        "event": "iteration_end", "iteration": 12345,
        "objective": 1.23e100, "delta_objective": -1.23e100,
        "jt_r_inf_norm": 1.23e-100, "step_norm": 1.23e-100,
        "step_scale": .5, "line_search_trials": 3,
    }])
    assert len(text.splitlines()[0]) < 95
    assert text.splitlines()[1].split() == [
        "12345", "iteration_end", "1.23e+100", "-1.23e+100",
        "1.23e-100", "1.23e-100", "5.00e-01", "3",
    ]
