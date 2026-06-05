from __future__ import annotations

import csv
from dataclasses import dataclass

import pytest

import numpy as np

from rei.core.trajectory import TrajectoryMap
from rei.optimize.builder import compile_nls_problem
from rei.optimize.plot import (
    TermAttrPlotSeries,
    collect_plot_series_from_compiled_term_attrs,
    collect_plot_series_from_term_attrs,
    collect_trajectory_derivative_plot_series,
    collect_trajectory_derivative_plot_series_from_term_attrs,
    plot_series,
    plot_term_attrs,
    write_plot_series_csv,
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

    def test_collect_plot_series_accepts_string_shorthand(self) -> None:
        dsl = {
            "time": {"N": 1, "dt": 0.5},
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
                    "attrs": {"plot": "joint_q"},
                }
            ],
        }

        def build_state(_x_all, *, pack=None, time=None, required=None):
            del pack, time
            out = {}
            if required is None:
                return out
            for key in required:
                if key.field == "q":
                    out[key] = np.array([float(key.k)], dtype=float)
                elif key.field == "q_J_q":
                    out[key] = np.array([[1.0]], dtype=float)
            return out

        runtime = compile_nls_problem(dsl, build_state=build_state)
        series = collect_plot_series_from_term_attrs(runtime)

        assert len(series) == 1
        assert series[0].name == "joint_q"
        assert series[0].field == "q"
        assert series[0].ks == (0, 1)

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
                                "k0": 0,
                                "k1": "last",
                                "stride": 2,
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
        assert s1.ks == (0, 2)
        assert np.allclose(s1.x, np.array([0.0, 0.4], dtype=float))
        assert np.allclose(s1.y, np.array([[0.0, -0.0], [2.0, -2.0]], dtype=float))

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

    def test_collect_plot_series_ignores_traj_derivative_specs(self) -> None:
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
                    "attrs": {
                        "plot": {
                            "type": "traj_derivative",
                            "name": "joint_qdot",
                            "derivative_order": 1,
                        }
                    },
                }
            ],
        }

        runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
        assert collect_plot_series_from_term_attrs(runtime, strict=True) == []

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

    def test_plot_series_supports_group_priorities(self) -> None:
        x = np.array([0.0, 0.1, 0.2], dtype=float)

        series = [
            TermAttrPlotSeries(
                term_index=0,
                term_name="torque",
                name="joint_torque",
                owner_type="total_joint",
                owner_name="robot",
                dtype="dynamics",
                field="torque",
                frame=None,
                rel_frame=None,
                ks=(0, 1, 2),
                x=x,
                y=np.array([[10.0], [11.0], [12.0]], dtype=float),
                x_axis="time",
            ),
            TermAttrPlotSeries(
                term_index=-2,
                term_name="trajectory",
                name="joint_qdot",
                owner_type="total_joint",
                owner_name="trajectory",
                dtype="coord",
                field="qdot",
                frame=None,
                rel_frame=None,
                ks=(0, 1, 2),
                x=x,
                y=np.array([[1.0], [2.0], [3.0]], dtype=float),
                x_axis="time",
            ),
            TermAttrPlotSeries(
                term_index=1,
                term_name="q",
                name="joint_q",
                owner_type="total_joint",
                owner_name="robot",
                dtype="coord",
                field="q",
                frame=None,
                rel_frame=None,
                ks=(0, 1, 2),
                x=x,
                y=np.array([[0.0], [0.5], [1.0]], dtype=float),
                x_axis="time",
            ),
        ]

        fig, axes, out = plot_series(
            series,
            subplot_by="name",
            group_priorities=("joint_q", "joint_qdot", "joint_qddot"),
            title="demo",
        )

        try:
            assert [s.name for s in out] == ["joint_torque", "joint_qdot", "joint_q"]
            axes_flat = np.asarray(axes, dtype=object).reshape(-1)
            titles = [str(ax_i.get_title()) for ax_i in axes_flat]
            assert titles == ["joint_q", "joint_qdot", "joint_torque"]
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_plot_series_group_priorities_require_subplot_by_name(self) -> None:
        series = [
            TermAttrPlotSeries(
                term_index=0,
                term_name="q",
                name="joint_q",
                owner_type="total_joint",
                owner_name="robot",
                dtype="coord",
                field="q",
                frame=None,
                rel_frame=None,
                ks=(0,),
                x=np.array([0.0], dtype=float),
                y=np.array([[0.0]], dtype=float),
                x_axis="index",
            )
        ]
        with pytest.raises(ValueError, match="subplot_by='name'"):
            _ = plot_series(series, group_priorities=("joint_q",))

    def test_write_plot_series_csv_writes_wide_format(self, tmp_path) -> None:
        x = np.array([0.0, 0.1], dtype=float)
        series = [
            TermAttrPlotSeries(
                term_index=7,
                term_name="demo",
                name="joint_q",
                owner_type="total_joint",
                owner_name="robot",
                dtype="coord",
                field="q",
                frame=None,
                rel_frame=None,
                ks=(0, 1),
                x=x,
                y=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
                x_axis="time",
                component_labels=("q0", "q1"),
            )
        ]

        out_path = write_plot_series_csv(series, tmp_path / "traj.csv")
        assert out_path.is_file()

        with out_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0]["x_axis"] == "time"
        assert rows[0]["k"] == "0"
        assert rows[0]["x"] == "0"
        assert rows[0]["q0"] == "1"
        assert rows[0]["q1"] == "2"
        assert rows[1]["k"] == "1"
        assert rows[1]["x"] == "0.1"
        assert rows[1]["q0"] == "3"
        assert rows[1]["q1"] == "4"

    def test_write_plot_series_csv_rejects_empty(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="no series"):
            _ = write_plot_series_csv([], tmp_path / "empty.csv")


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

    def test_collect_trajectory_derivative_plot_series_from_term_attrs(self) -> None:
        runtime = compile_nls_problem(
            {
                "variables": [{"name": "p", "dim": 2, "init": [1.0, 2.0]}],
                "terms": [
                    {
                        "expr": {
                            "type": "sub",
                            "a": {"type": "get_var", "var": "p"},
                            "b": {"type": "const", "var": "p", "value": [0.0, 0.0]},
                        },
                        "cost": {"type": "l2"},
                        "attrs": {
                            "plot": [
                                {
                                    "type": "traj_derivative",
                                    "name": "joint_qdot",
                                    "derivative_order": 1,
                                },
                                {
                                    "type": "traj_derivative",
                                    "name": "joint_qddot",
                                    "derivative_order": 2,
                                },
                            ]
                        },
                    }
                ],
            },
            build_state=lambda *_args, **_kwargs: {},
        )

        traj_d1 = TrajectoryMap(
            A=np.array(
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                ],
                dtype=float,
            ),
            b=np.zeros((3,), dtype=float),
            steps=3,
            q_dim=1,
        )
        traj_d2 = TrajectoryMap(
            A=np.array(
                [
                    [0.0, 1.0],
                    [0.0, 2.0],
                    [0.0, 3.0],
                ],
                dtype=float,
            ),
            b=np.zeros((3,), dtype=float),
            steps=3,
            q_dim=1,
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
            dt=0.5,
        )

        series = collect_trajectory_derivative_plot_series_from_term_attrs(compiled)
        assert [s.name for s in series] == ["joint_qdot", "joint_qddot"]
        assert np.allclose(series[0].y.reshape(-1), np.array([1.0, 2.0, 3.0], dtype=float))
        assert np.allclose(series[1].y.reshape(-1), np.array([2.0, 4.0, 6.0], dtype=float))

    def test_collect_plot_series_from_compiled_term_attrs_combines_series(self, tmp_path) -> None:
        dsl = {
            "time": {"N": 2, "dt": 0.1},
            "variables": [{"name": "p", "dim": 2, "init": [1.0, 2.0]}],
            "terms": [
                {
                    "expr": {
                        "type": "sub",
                        "a": {
                            "type": "get_state",
                            "key": {
                                "k": 0,
                                "owner_type": "total_joint",
                                "owner_name": "robot",
                                "dtype": "coord",
                                "field": "q",
                            },
                            "jac": {"var": "p"},
                        },
                        "b": {"type": "const", "var": "p", "value": [0.0, 0.0]},
                    },
                    "cost": {"type": "l2"},
                    "attrs": {
                        "plot": {
                            "type": "state_traj",
                            "name": "joint_q",
                            "owner_type": "total_joint",
                            "owner_name": "robot",
                            "dtype": "coord",
                            "field": "q",
                        }
                    },
                },
                {
                    "expr": {
                        "type": "sub",
                        "a": {"type": "get_var", "var": "p"},
                        "b": {"type": "const", "var": "p", "value": [0.0, 0.0]},
                    },
                    "cost": {"type": "l2"},
                    "attrs": {
                        "plot": {
                            "type": "traj_derivative",
                            "name": "joint_qdot",
                            "derivative_order": 1,
                        }
                    },
                },
            ],
        }

        def build_state(_x_all, *, pack=None, time=None, required=None):
            del pack, time
            out = {}
            if required is None:
                return out
            for key in required:
                if key.dtype == "coord" and key.field == "q":
                    out[key] = np.array(
                        [float(key.k) + 10.0, float(key.k) + 20.0],
                        dtype=float,
                    )
                    continue
                if key.field == "q_J_p":
                    out[key] = np.eye(2, dtype=float)
            return out

        runtime = compile_nls_problem(dsl, build_state=build_state)
        traj_d1 = TrajectoryMap(
            A=np.array(
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                ],
                dtype=float,
            ),
            b=np.zeros((3,), dtype=float),
            steps=3,
            q_dim=1,
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

        series = collect_plot_series_from_compiled_term_attrs(compiled)
        assert [s.name for s in series] == ["joint_q", "joint_qdot"]

        csv_path = write_plot_series_csv(series, tmp_path / "combined.csv")
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            header = next(csv.reader(f))
        assert "joint_q[0]" in header
        assert "joint_qdot" in header
