from __future__ import annotations

from datetime import datetime

import numpy as np

from rei.core.outcome import SolveOutcome, SolveStats
from rei.core.timing import TimingReport, TimingSpan
from rei.optimize.textlog import (
    build_solver_iter_logger,
    build_timestamped_log_path,
    compress_iter_history,
    format_solver_text_log,
    write_text_log,
)


def _build_dummy_outcome() -> SolveOutcome:
    return SolveOutcome(
        solution=np.array([1.0], dtype=float),
        stats=SolveStats(
            status="converged",
            iterations=3,
            initial_objective=4.0,
            objective=1.0,
            residual_norm=1.0,
            step_norm=0.1,
            message="ok",
        ),
        timing=TimingReport(
            total_seconds=1.0,
            spans=(TimingSpan(name="solve.setup", seconds=0.2, count=1),),
        ),
        meta={"solver": "gauss_newton"},
    )


def test_build_solver_iter_logger_and_compress_rows() -> None:
    opts, on_iter, history = build_solver_iter_logger(
        "gauss_newton",
        {
            "max_iters": 10,
            "verbose": False,
            "verbose_every": 2,
        },
        print_prefix="forward",
    )
    assert opts == {"max_iters": 10}

    on_iter(0, 1.0, 0.0)
    on_iter(0, 0.9, 0.2)
    on_iter(1, 0.5, 0.1)
    assert history == [(0, 1.0, 0.0), (0, 0.9, 0.2), (1, 0.5, 0.1)]

    rows = compress_iter_history(history)
    assert rows == [(0, 0.9, 0.2), (1, 0.5, 0.1)]


def test_format_solver_text_log_includes_sections() -> None:
    outcome = _build_dummy_outcome()
    text = format_solver_text_log(
        title="demo log",
        solver="gauss_newton",
        outcome=outcome,
        requested_options={"max_iters": 10},
        solve_options={"max_iters": 10},
        iter_history=[(0, 1.0, 0.0), (0, 0.9, 0.2), (1, 0.5, 0.1)],
        header={"model": "sample.json"},
        timestamp=datetime(2024, 1, 2, 3, 4, 5),
        timing_title="forward timing",
        extra_sections=[("extra", {"a": 1})],
    )

    assert "=== demo log ===" in text
    assert "timestamp=2024-01-02T03:04:05" in text
    assert "[solve.settings]" in text
    assert "[solve.result]" in text
    assert "[solve.iter_log]" in text
    assert "n_rows=2" in text
    assert "[extra]" in text


def test_build_timestamped_log_path_and_write(tmp_path) -> None:
    path = build_timestamped_log_path(
        tmp_path,
        prefix="run",
        timestamp=datetime(2024, 1, 2, 3, 4, 5),
    )
    assert path.name == "run_20240102_030405.txt"

    write_text_log(path, "hello")
    assert path.read_text(encoding="utf-8") == "hello\n"
