from __future__ import annotations

from dataclasses import dataclass

import pytest

import numpy as np

from rei.core.trajectory import TrajectoryMap
from rei.optimize.builder import compile_nls_problem
from rei.optimize.plot import (
    collect_plot_series_from_term_attrs,
    collect_trajectory_derivative_plot_series,
    plot_term_attrs,
)

class TestPlotTermAttrs:
    def test_collect_plot_series_infers_state_key_from_term_expr(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.5},
            "variables": [{"name": "q", "dim": 1, "init": [0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "name": "q0_eq",
                        "a": {
                            "type": "get_state",
                            "key": {
                                "k": 0,
                                "owner_type": "total_joint",
                                "owner_name": "robot",
                                "dtype": "coord",
                                "field": "q",
                            },
                            "jac": {"var": "q"},
                        },
                        "b": {"type": "const", "var": "q", "value": [0.0]},
                    },
                    "cost": {"type": "l2"},
                    "attrs": {
                        "plot": {
                            "type": "state_traj",
                            "name": "joint_q",
                        }
                    },
                }
            ],
        }

        def build_state(_x_all, *, pack=None, time=None, required=None):
            del pack, time
            out = {}
            if required is None:
                return out
            for key in required:
                if key.dtype != "coord":
                    continue
                if key.field == "q":
                    out[key] = np.array([1.0 + float(key.k)], dtype=float)
                    continue
                if key.field == "q_J_q":
                    out[key] = np.array([[1.0]], dtype=float)
            return out

        runtime = compile_nls_problem(dsl, build_state=build_state)
        series = collect_plot_series_from_term_attrs(runtime)
        assert len(series) == 1

        s = series[0]
        assert s.name == "joint_q"
        assert s.term_name == "q0_eq"
        assert s.owner_type == "total_joint"
        assert s.owner_name == "robot"
        assert s.dtype == "coord"
        assert s.field == "q"
        assert s.ks == (0, 1, 2)
        assert s.x_axis == "time"
        assert np.allclose(s.x, np.array([0.0, 0.5, 1.0], dtype=float))
        assert np.allclose(s.y, np.array([[1.0], [2.0], [3.0]], dtype=float))

    def test_collect_plot_series_supports_explicit_key_and_components(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.2},
            "variables": [{"name": "q", "dim": 2, "init": [0.0, 0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "name": "q0_eq",
                        "a": {
                            "type": "get_state",
                            "key": {
                                "k": 0,
                                "owner_type": "total_joint",
                                "owner_name": "robot",
                                "dtype": "coord",
                                "field": "q",
                            },
                            "jac": {"var": "q"},
                        },
                        "b": {"type": "const", "var": "q", "value": [0.0, 0.0]},
                    },
                    "cost": {"type": "l2"},
                    "attrs": {
                        "plot": [
                            {
                                "type": "state_traj",
                                "name": "joint_q",
                                "key": {
                                    "owner_type": "total_joint",
                                    "owner_name": "robot",
                                    "dtype": "coord",
                                    "field": "q",
                                },
                                "ks": [0, 2],
                                "components": ["q0", "q1"],
                            },
                            {
                                "type": "state_traj",
                                "name": "joint_q_tail",
                                "owner_type": "total_joint",
                                "owner_name": "robot",
                                "dtype": "coord",
                                "field": "q",
                                "k0": 1,
                                "k1": "last",
                            },
                        ]
                    },
                }
            ],
        }

        def build_state(_x_all, *, pack=None, time=None, required=None):
            del pack, time
            out = {}
            if required is None:
                return out
            for key in required:
                if key.dtype != "coord":
                    continue
                if key.field == "q":
                    k = float(key.k)
                    out[key] = np.array([k, -k], dtype=float)
                    continue
                if key.field == "q_J_q":
                    out[key] = np.eye(2, dtype=float)
            return out

        runtime = compile_nls_problem(dsl, build_state=build_state)
        series = collect_plot_series_from_term_attrs(runtime)
        assert len(series) == 2

        s0 = series[0]
        assert s0.name == "joint_q"
        assert s0.ks == (0, 2)
        assert s0.component_labels == ("q0", "q1")
        assert s0.line_label(0) == "q0"
        assert s0.line_label(1) == "q1"
        assert np.allclose(s0.x, np.array([0.0, 0.4], dtype=float))
        assert np.allclose(s0.y, np.array([[0.0, -0.0], [2.0, -2.0]], dtype=float))

        s1 = series[1]
        assert s1.name == "joint_q_tail"
        assert s1.ks == (1, 2)
        assert np.allclose(s1.x, np.array([0.2, 0.4], dtype=float))
        assert np.allclose(s1.y, np.array([[1.0, -1.0], [2.0, -2.0]], dtype=float))

    def test_collect_plot_series_handles_unknown_plot_type_by_strict_flag(self) -> None:
        dsl = {
            "variables": [{"name": "x", "dim": 1, "init": [0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "a": {"type": "get_var", "var": "x"},
                        "b": {"type": "const", "var": "x", "value": [0.0]},
                    },
                    "cost": {"type": "l2"},
                    "attrs": {"plot": {"type": "unknown_kind"}},
                }
            ],
        }

        runtime = compile_nls_problem(
            dsl,
            build_state=lambda *_args, **_kwargs: {},
        )

        assert collect_plot_series_from_term_attrs(runtime, strict=False) == []
        with pytest.raises(ValueError, match="unsupported plot type"):
            _ = collect_plot_series_from_term_attrs(runtime, strict=True)

    def test_plot_term_attrs_supports_subplot_by_name(self) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.1},
            "variables": [{"name": "x", "dim": 1, "init": [0.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "name": "x_eq",
                        "a": {
                            "type": "get_state",
                            "key": {
                                "k": 0,
                                "owner_type": "total_joint",
                                "owner_name": "robot",
                                "dtype": "coord",
                                "field": "q",
                            },
                            "jac": {"var": "x"},
                        },
                        "b": {"type": "const", "var": "x", "value": [0.0]},
                    },
                    "cost": {"type": "l2"},
                    "attrs": {
                        "plot": [
                            {
                                "type": "state_traj",
                                "name": "joint_q",
                                "owner_type": "total_joint",
                                "owner_name": "robot",
                                "dtype": "coord",
                                "field": "q",
                            },
                            {
                                "type": "state_traj",
                                "name": "joint_torque",
                                "owner_type": "total_joint",
                                "owner_name": "robot",
                                "dtype": "dynamics",
                                "field": "torque",
                            },
                        ]
                    },
                }
            ],
        }

        def build_state(_x_all, *, pack=None, time=None, required=None):
            del pack, time
            out = {}
            if required is None:
                return out
            for key in required:
                if key.dtype == "coord" and key.field == "q":
                    out[key] = np.array([float(key.k)], dtype=float)
                    continue
                if key.dtype == "dynamics" and key.field == "torque":
                    out[key] = np.array([10.0 + float(key.k)], dtype=float)
                    continue
                if key.field == "q_J_x" or key.field == "torque_J_x":
                    out[key] = np.array([[1.0]], dtype=float)
            return out

        runtime = compile_nls_problem(dsl, build_state=build_state)
        fig, axes, series = plot_term_attrs(runtime, subplot_by="name", title="demo")
        assert len(series) == 2

        try:
            axes_flat = np.asarray(axes, dtype=object).reshape(-1)
            assert len(axes_flat) == 2
            titles = [str(ax_i.get_title()) for ax_i in axes_flat]
            assert "joint_q" in titles
            assert "joint_torque" in titles
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)


class TestTrajectoryDerivativePlotSeries:
    @staticmethod
    def _build_runtime_with_p(*, p_init: list[float]) -> object:
        dsl = {
            "variables": [{"name": "p", "dim": len(p_init), "init": p_init}],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "a": {"type": "get_var", "var": "p"},
                        "b": {"type": "const", "var": "p", "value": [0.0 for _ in p_init]},
                    },
                    "cost": {"type": "l2"},
                }
            ],
        }
        return compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})

    def test_collect_trajectory_derivative_plot_series_evaluates_maps(self) -> None:
        runtime = self._build_runtime_with_p(p_init=[1.0, 2.0])

        traj_d1 = TrajectoryMap(
            A=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [3.0, 0.0],
                    [0.0, 3.0],
                ],
                dtype=float,
            ),
            b=np.zeros((6,), dtype=float),
            steps=3,
            q_dim=2,
        )
        traj_d2 = TrajectoryMap(
            A=np.array(
                [
                    [1.0, 1.0],
                    [1.0, -1.0],
                    [1.0, 1.0],
                    [1.0, -1.0],
                    [1.0, 1.0],
                    [1.0, -1.0],
                ],
                dtype=float,
            ),
            b=np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=float),
            steps=3,
            q_dim=2,
        )

        @dataclass(frozen=True)
        class _CompiledStub:
            runtime: object
            trajectory_derivative_maps: dict[int, TrajectoryMap]
            p_var: str
            dt: float

        compiled = _CompiledStub(
            runtime=runtime,
            trajectory_derivative_maps={1: traj_d1, 2: traj_d2},
            p_var="p",
            dt=0.2,
        )

        series = collect_trajectory_derivative_plot_series(compiled, derivative_orders=(1, 2))
        assert len(series) == 2
        assert [s.name for s in series] == ["joint_qdot", "joint_qddot"]
        assert [s.x_axis for s in series] == ["time", "time"]
        assert np.allclose(series[0].x, np.array([0.0, 0.2, 0.4], dtype=float))
        assert np.allclose(series[1].x, np.array([0.0, 0.2, 0.4], dtype=float))
        assert np.allclose(
            series[0].y,
            np.array(
                [
                    [1.0, 2.0],
                    [2.0, 4.0],
                    [3.0, 6.0],
                ],
                dtype=float,
            ),
        )
        assert np.allclose(
            series[1].y,
            np.array(
                [
                    [3.0, -1.0],
                    [4.0, 0.0],
                    [5.0, 1.0],
                ],
                dtype=float,
            ),
        )

    def test_collect_trajectory_derivative_plot_series_strict_missing_order(self) -> None:
        runtime = self._build_runtime_with_p(p_init=[1.0, 2.0])
        traj_d1 = TrajectoryMap(
            A=np.eye(6, 2, dtype=float),
            b=np.zeros((6,), dtype=float),
            steps=3,
            q_dim=2,
        )

        @dataclass(frozen=True)
        class _CompiledStub:
            runtime: object
            trajectory_derivative_maps: dict[int, TrajectoryMap]
            p_var: str
            dt: float

        compiled = _CompiledStub(
            runtime=runtime,
            trajectory_derivative_maps={1: traj_d1},
            p_var="p",
            dt=0.1,
        )

        out = collect_trajectory_derivative_plot_series(
            compiled,
            derivative_orders=(2,),
            strict=False,
        )
        assert out == []

        with pytest.raises(ValueError, match="order=2"):
            _ = collect_trajectory_derivative_plot_series(
                compiled,
                derivative_orders=(2,),
                strict=True,
            )
