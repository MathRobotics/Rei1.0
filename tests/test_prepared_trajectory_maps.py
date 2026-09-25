from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import rei.core.bspline as bspline
from rei.core.trajectory import TrajectoryMap
from rei.core.trajectory_dsl import build_trajectory_maps_with_derivatives
from rei.optimize.builder import compile_nls_problem
from rei.optimize.dsl.trajectory_compile import prepare_trajectory_problem_dsl
from rei.optimize_backends.trajectory_adapter import compile_trajectory_problem_with_adapter
from rei.optimize.dsl.environment import DslBuildEnv
from rei.core.time_grid import TimeGrid
from rei.core.expr.types import VariablePack
from rei.core.expr.registry import ExprRegister


def dsl(q_dim=1):
    return {
        "time": {"N": 5, "dt": .2},
        "trajectory": {"type": "bspline", "degree": 3, "num_ctrl_points": 5, "q_dim": q_dim},
        "variables": [{"name": "p", "dim": 5*q_dim, "init": [0.]* (5*q_dim)}],
        "terms": [{"expr": {"type": "get_traj_var", "var": "p", "derivative_order": order,
                            "derivative_wrt": "time"}, "cost": {"type": "l2"}}
                  for order in range(4)],
    }


def maps_for(spec, max_order=3):
    return build_trajectory_maps_with_derivatives(spec["trajectory"], max_derivative_order=max_order,
                                                default_steps=6, derivative_wrt="time", default_dt=.2)


def observe_basis(monkeypatch):
    calls = []
    original = bspline.bspline_basis_matrix

    def observed(**kwargs):
        calls.append(kwargs["degree"])
        return original(**kwargs)

    monkeypatch.setattr(bspline, "bspline_basis_matrix", observed)
    return calls


class Adapter:
    def infer_model_dof(self, model):
        return 1

    def infer_model_order(self, model):
        return 4

    def build_state_builder(self, *, prepared, **kwargs):
        return SimpleNamespace(trajectory_map=prepared.trajectory_map,
                               trajectory_derivative_maps=prepared.trajectory_derivative_maps,
                               build_state=lambda *args, **kwargs: {})

    def validate_runtime(self, **kwargs):
        pass


@pytest.mark.parametrize("provided", [False, True])
def test_prepared_maps_shared_by_graph_and_state_builder(monkeypatch, provided):
    spec = dsl()
    external = maps_for(spec) if provided else None
    calls = observe_basis(monkeypatch)
    compiled = compile_trajectory_problem_with_adapter(spec, model=None, data=None,
                                                     adapter=Adapter(), trajectory_maps=external)
    assert calls == ([] if provided else [3, 2, 1, 0])
    assert compiled.prepared.trajectory_map is compiled.prepared.trajectory_derivative_maps[0]
    assert compiled.state_builder.trajectory_map is compiled.prepared.trajectory_map
    for order, (expr, _) in enumerate(compiled.runtime.problem.terms):
        assert expr.trajectory is compiled.prepared.trajectory_derivative_maps[order]
        assert expr.trajectory is compiled.state_builder.trajectory_derivative_maps[order]
        if external is not None:
            assert expr.trajectory is external[order]
    assert "steps" not in spec["trajectory"]  # preparation must not mutate caller DSL


@pytest.mark.parametrize("missing", [{3}, {1, 3}])
def test_partial_maps_only_evaluate_missing_orders(monkeypatch, missing):
    spec = dsl()
    reference = maps_for(spec)
    existing = {r: m for r, m in enumerate(reference) if r not in missing}
    calls = observe_basis(monkeypatch)
    prepared = prepare_trajectory_problem_dsl(spec, model_order=4, trajectory_maps=existing)
    assert calls == [3-r for r in sorted(missing)]
    for r, actual in prepared.trajectory_derivative_maps.items():
        np.testing.assert_allclose(np.asarray(actual.A), np.asarray(reference[r].A))
        if r in existing:
            assert actual is existing[r]


def test_cache_growth_retains_existing_map_identity(monkeypatch):
    spec = dsl()["trajectory"]
    env = DslBuildEnv(pack=VariablePack([]), time=TimeGrid(N=5, dt=.2), expr_register=ExprRegister())
    first = env.resolve_trajectory_maps_with_derivatives(spec, max_derivative_order=2, derivative_wrt="time")
    calls = observe_basis(monkeypatch)
    expanded = env.resolve_trajectory_maps_with_derivatives(spec, max_derivative_order=3, derivative_wrt="time")
    assert calls == [0]
    assert all(first[r] is expanded[r] for r in range(3))
    assert env.resolve_trajectory_map(spec) is first[0]
    env.time.update(dt=.4)
    changed = env.resolve_trajectory_map_with_derivative(spec, derivative_order=3, derivative_wrt="time")
    np.testing.assert_allclose(np.asarray(changed.A), np.asarray(expanded[3].A)/8)
    assert env.resolve_trajectory_map(spec) is first[0]


def test_parameter_derivatives_do_not_reuse_time_derivatives(monkeypatch):
    spec = dsl()
    external = maps_for(spec)
    spec["terms"][1]["expr"]["derivative_wrt"] = "parameter"
    calls = observe_basis(monkeypatch)
    runtime = compile_nls_problem(spec, build_state=lambda *args, **kwargs: {}, trajectory_maps=external)
    expr = runtime.problem.terms[1][0]
    assert expr.trajectory is not external[1]
    # Horizon is 1 second here; still separate identity/caching by units.
    assert calls == [2]
    assert runtime.problem.terms[0][0].trajectory is external[0]


@pytest.mark.parametrize("backend", ["kots", "pinocchio"])
def test_public_backend_and_ioc_accept_precomputed_maps(monkeypatch, backend):
    from rei.optimize_backends.trajectory_ioc import compile_trajectory_ioc_problem

    root = Path(__file__).resolve().parents[1] / "examples/models"
    if backend == "kots":
        Kots = pytest.importorskip("robokots.kots").Kots
        model = Kots.from_json_file(str(root / "planar2.json"), order=4)
        data = None
    else:
        pin = pytest.importorskip("pinocchio")
        model = pin.buildModelFromUrdf(str(root / "planar2.urdf"))
        data = model.createData()
    spec = dsl(q_dim=2)
    external = maps_for(spec)
    calls = observe_basis(monkeypatch)
    compiled = compile_trajectory_ioc_problem(spec, backend=backend, model=model, data=data,
                                            max_derivative_order=3, trajectory_maps=external)
    assert calls == []
    for r, (expr, _) in enumerate(compiled.runtime.problem.terms):
        assert expr.trajectory is external[r]
        assert compiled.trajectory_derivative_maps[r] is external[r]
    reference = compile_trajectory_ioc_problem(spec, backend=backend, model=model, data=data,
                                             max_derivative_order=3)
    compiled.runtime.pack.set(np.linspace(-.2, .5, 10))
    reference.runtime.pack.set(np.linspace(-.2, .5, 10))
    for actual, expected in zip(compiled.runtime.linearize(), reference.runtime.linearize()):
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("case", ["steps", "q_dim", "p_dim", "mixed", "order", "missing_zero"])
def test_external_map_validation(case):
    spec = dsl()
    maps = dict(enumerate(maps_for(spec)))
    if case == "steps":
        spec["trajectory"]["steps"] = 7
    elif case == "q_dim":
        spec["trajectory"]["q_dim"] = 2
    elif case == "p_dim":
        spec["trajectory"]["num_ctrl_points"] = 6
    elif case == "mixed":
        maps[1] = TrajectoryMap(A=np.eye(6), b=np.zeros(6), steps=6, q_dim=1)
    elif case == "order":
        maps["1"] = maps[1]
    else:
        del maps[0]
    with pytest.raises(ValueError):
        prepare_trajectory_problem_dsl(spec, model_order=4, trajectory_maps=maps)


@pytest.mark.parametrize("kind", ["linear", "nonuniform_bspline"])
def test_finite_difference_cache_extension_matches_cold_build(kind, monkeypatch):
    spec = dsl()
    if kind == "linear":
        spec["trajectory"] = {"type": "linear", "steps": 6, "q_dim": 1,
                              "A": np.arange(30).reshape(6, 5).tolist(), "b": [0., 1., 3., 4., 8., 9.]}
    else:
        spec["trajectory"]["u_samples"] = [0., .1, .2, .5, .7, 1.]
    reference = maps_for(spec)
    existing = dict(enumerate(maps_for(spec, max_order=2)))
    calls = observe_basis(monkeypatch)
    result = build_trajectory_maps_with_derivatives(spec["trajectory"], max_derivative_order=3,
        derivative_wrt="time", default_steps=6, default_dt=.2, existing_maps=existing)
    assert calls == []
    assert all(result[r] is existing[r] for r in range(3))
    np.testing.assert_allclose(np.asarray(result[3].A), np.asarray(reference[3].A))
    np.testing.assert_allclose(result[3].b, reference[3].b)


def test_default_time_grid_is_shared_with_expression_graph(monkeypatch):
    spec = dsl()
    del spec["time"]
    calls = observe_basis(monkeypatch)
    compiled = compile_trajectory_problem_with_adapter(spec, model=None, data=None,
        adapter=Adapter(), default_steps=6, default_dt=.2)
    assert calls == [3, 2, 1, 0]
    assert compiled.runtime.time.dt == .2
    assert compiled.runtime.time.N == 5
    for order, (expr, _) in enumerate(compiled.runtime.problem.terms):
        assert expr.trajectory is compiled.prepared.trajectory_derivative_maps[order]


def test_spec_api_and_nested_trajectory_share_maps(monkeypatch):
    from rei.optimize.builder import compile_nls_problem_spec

    trajectory = {"type": "bspline", "bspline": {"degree": 3, "num_ctrl_points": 5, "q_dim": 1}}
    maps = build_trajectory_maps_with_derivatives(trajectory, max_derivative_order=3,
        derivative_wrt="time", default_steps=6, default_dt=.2)
    calls = observe_basis(monkeypatch)
    runtime = compile_nls_problem_spec({
        "time": {"N": 5, "dt": .2}, "trajectory": trajectory,
        "opt_vals": {"trajectory_params": {"dim": 5, "init": {"fill": 0.}}},
        "terms": [{"name": "jerk", "quantity": {"name": "joint_angles", "derivative_order": 3,
                                                 "derivative_wrt": "time"}}],
    }, build_state=lambda *args, **kwargs: {}, trajectory_maps=maps)
    assert calls == []
    # Even a stacked series should reuse the exact map on every time slice.
    term = runtime.problem.terms[0][0]
    parts = getattr(term, "parts", [term])
    assert all(part.trajectory is maps[3] for part in parts)
