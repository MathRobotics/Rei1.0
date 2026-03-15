from __future__ import annotations

import numpy as np
import pytest

from rei.core.outcome import SolveOutcome, SolveStats
from rei.core.timing import TimingReport
from rei.equations.report import build_ioc_log_sections, format_ioc_report
from rei.equations.stationarity import StationarityTermContribution


def _dummy_simplex_outcome() -> SolveOutcome:
    return SolveOutcome(
        solution=np.array([0.7, 0.3], dtype=float),
        stats=SolveStats(
            status="converged",
            iterations=5,
            objective=1e-6,
        ),
        timing=TimingReport(total_seconds=0.05, spans=tuple()),
        meta={"solver": "simplex_min_norm", "method": "qr_nullspace"},
    )


def test_format_ioc_report_basic() -> None:
    text = format_ioc_report(
        active_mode="gradient",
        active_idx=[0, 1],
        active_grad_idx=[0, 1],
        active_res_idx=[1],
        term_indices=[3, 5],
        w_true=np.array([0.6, 0.4], dtype=float),
        w_hat=np.array([0.7, 0.3], dtype=float),
        ioc_identifiable=True,
        ikkt_ok=False,
        ikkt_residual=2.0e-3,
        ikkt_tol=1.0e-6,
        simplex_out=_dummy_simplex_outcome(),
    )
    assert "IOC:" in text
    assert "active_mode=gradient" in text
    assert "term_indices=[3, 5]" in text
    assert "L1 error=" in text
    assert "simplex:" in text


def test_format_ioc_report_stationarity_terms() -> None:
    contrib = [
        StationarityTermContribution(
            term_index=4,
            name="demo_term",
            attrs={"is_constraint": False},
            gradient=np.array([1.0, -2.0], dtype=float),
            cost_name="scalar_weight",
            weighted_residual_norm=0.2,
        )
    ]
    text = format_ioc_report(
        active_mode="residual",
        active_idx=[0],
        active_grad_idx=[0],
        active_res_idx=[0],
        term_indices=[4],
        w_hat=np.array([1.0], dtype=float),
        contributions=contrib,
        include_stationarity_terms=True,
    )
    assert "Stationarity terms:" in text
    assert "term[4]" in text
    assert "cost=scalar_weight" in text
    assert "w_hat=1.000e+00" in text

    with pytest.raises(ValueError, match="contributions is required"):
        _ = format_ioc_report(
            active_mode="residual",
            active_idx=[0],
            active_grad_idx=[0],
            active_res_idx=[0],
            term_indices=[4],
            w_hat=np.array([1.0], dtype=float),
            include_stationarity_terms=True,
        )


def test_build_ioc_log_sections() -> None:
    out = _dummy_simplex_outcome()
    contrib = [
        StationarityTermContribution(
            term_index=3,
            name="q_terminal",
            attrs={"is_constraint": False},
            gradient=np.array([1.0, 0.0], dtype=float),
            cost_name="scalar_weight",
            weighted_residual_norm=0.3,
        ),
        StationarityTermContribution(
            term_index=5,
            name="torque_regularization",
            attrs={"is_constraint": False},
            gradient=np.array([0.0, 2.0], dtype=float),
            cost_name="huber",
            weighted_residual_norm=0.0,
        ),
    ]
    sections = build_ioc_log_sections(
        callback_rows=12,
        active_mode="gradient",
        active_idx=[0, 1],
        active_grad_idx=[0, 1],
        active_res_idx=[1],
        term_indices=[3, 5],
        w_true=np.array([0.6, 0.4], dtype=float),
        w_hat=np.array([0.7, 0.3], dtype=float),
        ioc_identifiable=True,
        ikkt_ok=True,
        ikkt_residual=1.0e-7,
        ikkt_tol=1.0e-6,
        ioc_max_iters=1000,
        simplex_out=out,
        contributions=contrib,
    )
    names = [name for name, _body in sections]
    assert "solve.iter_meta" in names
    assert "ioc.settings" in names
    assert "ioc.result" in names
    assert "ioc.objective_terms" in names
    assert "simplex.result" in names

    objective_terms = dict(sections)["ioc.objective_terms"]
    assert any("q_terminal" in line for line in objective_terms)
    assert any("cost=scalar_weight" in line for line in objective_terms)
    assert any("cost=huber" in line for line in objective_terms)
