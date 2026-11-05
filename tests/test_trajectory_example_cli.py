"""Exercise trajectory example solver selection and the written trial histories."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize("solver,globalization,history_flag,event,strategy", [
    ("levenberg_marquardt", None, "--trial-history", "lm_trial", "dense"),
    ("gauss_newton_krylov", None, "--line-search-history", "line_search_trial", "mul"),
    ("gauss_newton_krylov", "trust_region", "--trial-history", "trust_region_trial", "mul"),
])
def test_numpy_example_routes_solver_and_trial_history(
    monkeypatch, tmp_path, capsys, solver, globalization, history_flag, event, strategy,
):
    pytest.importorskip("robokots.kots")
    root = Path(__file__).resolve().parents[1]
    module_spec = importlib.util.spec_from_file_location(
        "rei_trajectory_example", root / "examples/robokots_trajectory_dynamics.py")
    example = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(example)
    problem = example.load_problem_spec_toml(example._SPEC_PATH)
    problem["time"].update(N=5, dt=.2)
    problem["trajectory"]["num_ctrl_points"] = 6
    monkeypatch.setattr(example, "load_problem_spec_toml", lambda _: problem)
    path = tmp_path / "trials.jsonl"
    argv = ["example", "--backend", "numpy", "--solver", solver,
            "--quiet", history_flag, str(path)]
    if globalization:
        argv += ["--globalization", globalization]
    monkeypatch.setattr(sys, "argv", argv)
    example.main()
    output = capsys.readouterr().out
    assert "status=converged" in output
    assert f"'jacobian_strategy': '{strategy}'" in output
    events = [json.loads(line)["event"] for line in path.read_text().splitlines()]
    assert event in events
