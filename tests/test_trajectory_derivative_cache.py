from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

import rei.optimize.dsl.environment as dsl_environment
from rei.core.expr.registry import ExprRegister
from rei.core.expr.types import VariablePack
from rei.core.time_grid import TimeGrid
from rei.optimize.builder import compile_nls_problem
from rei.optimize.dsl.environment import DslBuildEnv


def _count_derivative_map_builds(monkeypatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    original = dsl_environment.build_trajectory_maps_with_derivatives

    def counted(traj_dsl: Mapping[str, Any], **kwargs):
        calls.append({"trajectory": traj_dsl, **kwargs})
        return original(traj_dsl, **kwargs)

    monkeypatch.setattr(dsl_environment, "build_trajectory_maps_with_derivatives", counted)
    return calls


def _build_env(*, steps: int = 6, dt: float = 0.1) -> DslBuildEnv:
    return DslBuildEnv(
        pack=VariablePack(vars=[]),
        time=TimeGrid(N=steps - 1, dt=dt),
        expr_register=ExprRegister(),
    )


def test_compile_reuses_one_qdot_map_for_509_vstack_parts(monkeypatch) -> None:
    calls = _count_derivative_map_builds(monkeypatch)
    steps = 509
    trajectory = {
        "type": "bspline",
        "var": "p",
        "degree": 5,
        "num_ctrl_points": 6,
        "q_dim": 1,
    }
    qdot_parts = [
        {
            "type": "get_traj_var",
            "name": f"qdot_{k}",
            "var": "p",
            "derivative_order": 1,
            "derivative_wrt": "time",
            "k": k,
        }
        for k in range(steps)
    ]
    dsl = {
        "time": {"N": steps - 1, "dt": 1.0 / 120.0},
        "trajectory": trajectory,
        "variables": [{"name": "p", "dim": 6, "init": np.zeros(6).tolist()}],
        "terms": [
            {
                "expr": {"type": "vstack", "name": "all_qdot", "parts": qdot_parts},
                "cost": {"type": "l2"},
            }
        ],
    }

    compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})

    assert len(calls) == 1
    assert calls[0]["max_derivative_order"] == 1
    assert calls[0]["derivative_wrt"] == "time"
    assert calls[0]["default_steps"] == steps


def test_compile_routes_max_derivative_expressions_through_cache(monkeypatch) -> None:
    calls = _count_derivative_map_builds(monkeypatch)
    dsl = {
        "time": {"N": 3, "dt": 0.1},
        "trajectory": {
            "type": "bspline",
            "var": "p",
            "degree": 2,
            "num_ctrl_points": 3,
            "q_dim": 1,
        },
        "variables": [{"name": "p", "dim": 3, "init": [0.0, 0.0, 0.0]}],
        "terms": [
            {
                "expr": {
                    "type": "vstack",
                    "parts": [
                        {
                            "type": "get_traj_var",
                            "var": "p",
                            "max_derivative_order": 2,
                            "derivative_wrt": "time",
                            "k": k,
                        }
                        for k in range(4)
                    ],
                },
                "cost": {"type": "l2"},
            }
        ],
    }

    compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})

    assert len(calls) == 1
    assert calls[0]["max_derivative_order"] == 2


def test_derivative_cache_reuses_max_order_maps_and_returns_fresh_lists(monkeypatch) -> None:
    calls = _count_derivative_map_builds(monkeypatch)
    env = _build_env()
    trajectory = {"type": "bspline", "degree": 3, "num_ctrl_points": 4}

    first = env.resolve_trajectory_maps_with_derivatives(
        trajectory,
        max_derivative_order=3,
        derivative_wrt="time",
        default_q_dim=2,
    )
    second = env.resolve_trajectory_maps_with_derivatives(
        trajectory,
        max_derivative_order=3,
        derivative_wrt="time",
        default_q_dim=2,
    )
    lower = env.resolve_trajectory_maps_with_derivatives(
        trajectory,
        max_derivative_order=2,
        derivative_wrt="time",
        default_q_dim=2,
    )
    first_derivative = env.resolve_trajectory_map_with_derivative(
        trajectory,
        derivative_order=1,
        derivative_wrt="time",
        default_q_dim=2,
    )

    assert len(calls) == 1
    assert first is not second
    assert all(a is b for a, b in zip(first, second, strict=True))
    assert all(a is b for a, b in zip(first[:3], lower, strict=True))
    assert first_derivative is first[1]


def test_derivative_cache_key_separates_inputs_and_tracks_spec_mutation(monkeypatch) -> None:
    calls = _count_derivative_map_builds(monkeypatch)
    env = _build_env(steps=5, dt=0.1)
    trajectory = {"type": "bspline", "degree": 2, "num_ctrl_points": 3}

    time_map = env.resolve_trajectory_map_with_derivative(
        trajectory, derivative_order=1, derivative_wrt="time", default_q_dim=1
    )
    assert (
        env.resolve_trajectory_map_with_derivative(
            trajectory, derivative_order=1, derivative_wrt="time", default_q_dim=1
        )
        is time_map
    )
    assert len(calls) == 1

    parameter_map = env.resolve_trajectory_map_with_derivative(
        trajectory, derivative_order=1, derivative_wrt="u", default_q_dim=1
    )
    assert (
        env.resolve_trajectory_map_with_derivative(
            trajectory, derivative_order=1, derivative_wrt="parameter", default_q_dim=1
        )
        is parameter_map
    )
    assert len(calls) == 2

    env.time.update(dt=0.2)
    different_dt = env.resolve_trajectory_map_with_derivative(
        trajectory, derivative_order=1, derivative_wrt="time", default_q_dim=1
    )
    assert different_dt is not time_map
    assert len(calls) == 3
    assert (
        env.resolve_trajectory_map_with_derivative(
            trajectory, derivative_order=1, derivative_wrt="param", default_q_dim=1
        )
        is parameter_map
    )
    assert len(calls) == 3

    env.time.update(N=5)
    different_steps = env.resolve_trajectory_map_with_derivative(
        trajectory, derivative_order=1, derivative_wrt="time", default_q_dim=1
    )
    assert different_steps is not different_dt
    assert len(calls) == 4

    different_q_dim = env.resolve_trajectory_map_with_derivative(
        trajectory, derivative_order=1, derivative_wrt="time", default_q_dim=2
    )
    assert different_q_dim is not different_steps
    assert len(calls) == 5

    trajectory["degree"] = 1
    mutated_spec = env.resolve_trajectory_map_with_derivative(
        trajectory, derivative_order=1, derivative_wrt="time", default_q_dim=2
    )
    assert mutated_spec is not different_q_dim
    assert len(calls) == 6

    equivalent_spec = dict(trajectory)
    assert (
        env.resolve_trajectory_map_with_derivative(
            equivalent_spec, derivative_order=1, derivative_wrt="time", default_q_dim=2
        )
        is mutated_spec
    )
    assert len(calls) == 6

    second_derivative = env.resolve_trajectory_map_with_derivative(
        equivalent_spec, derivative_order=2, derivative_wrt="time", default_q_dim=2
    )
    assert second_derivative is not mutated_spec
    assert len(calls) == 7
