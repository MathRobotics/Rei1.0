"""TOML model metadata is resolved independently, then loaded by the backend."""
import copy
from pathlib import Path
import shutil

import numpy as np
import pytest

from rei import load_problem_spec_toml
from rei.optimize.dsl import problem_spec_to_dsl
from rei.backends.state.robotics.urdf import load_joint_locks_toml


ROOT = Path(__file__).resolve().parents[1]


def problem():
    spec = load_problem_spec_toml(ROOT / "examples/spec/robokots_traj_dynamics_d12.toml")
    spec["time"].update(N=3, dt=.2)
    spec["trajectory"]["num_ctrl_points"] = 6
    return spec


def test_path_is_relative_to_toml_not_working_directory(tmp_path, monkeypatch):
    specs = tmp_path / "specs"
    specs.mkdir()
    config = specs / "problem.toml"
    config.write_text('[model]\nfile = "../robot.urdf"\n[model.locked_joints]\njoint2 = 0.2\n')
    monkeypatch.chdir(ROOT)
    loaded = load_problem_spec_toml(config)
    assert loaded["model"] == {"file": str(tmp_path / "robot.urdf"), "locked_joints": {"joint2": .2}}
    # Parsing metadata does not load or require the model file.
    assert load_joint_locks_toml(config) == {"joint2": .2}


def test_mapping_preserves_metadata_without_mutating_input():
    spec = {"model": {"file": "robot.urdf", "locked_joints": {"joint2": .2}}}
    original = copy.deepcopy(spec)
    dsl = problem_spec_to_dsl(spec)
    assert dsl["model"] == original["model"]
    dsl["model"]["locked_joints"]["joint2"] = .4
    assert spec == original


@pytest.mark.parametrize("model", [None, [], {"file": 3}, {"file": " "}, {"path": "robot.urdf"}])
def test_invalid_model_metadata(model):
    with pytest.raises(ValueError):
        problem_spec_to_dsl({"model": model})


@pytest.mark.parametrize("extension", ["urdf", "json"])
@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_kots_file_and_explicit_model_match(extension, backend, tmp_path, monkeypatch):
    Kots = pytest.importorskip("robokots.kots").Kots
    from rei.optimize_backends.kots import compile_kots_trajectory_problem

    path = ROOT / f"examples/models/planar2.{extension}"
    shutil.copy(path, tmp_path / path.name)
    config = tmp_path / "model.toml"
    config.write_text(f'[model]\nfile = "{path.name}"\n')
    metadata = load_problem_spec_toml(config)["model"]
    monkeypatch.chdir(ROOT.parent)
    spec = problem()
    spec["model"] = metadata
    original = copy.deepcopy(spec)
    loaded = compile_kots_trajectory_problem(spec, kots_backend=backend)
    constructor = Kots.from_urdf_file if extension == "urdf" else Kots.from_json_file
    explicit = compile_kots_trajectory_problem(spec, model=constructor(str(path), order=3, backend=backend), kots_backend=backend)
    point = np.random.default_rng(9).normal(scale=.1, size=loaded.runtime.pack.n_total)
    loaded.runtime.pack.set(point)
    explicit.runtime.pack.set(point)
    for actual, expected in zip(loaded.runtime.linearize(), explicit.runtime.linearize(), strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert loaded.model_order == 3
    assert spec == original


def test_explicit_model_overrides_file_and_locks():
    Kots = pytest.importorskip("robokots.kots").Kots
    from rei.optimize_backends.kots import compile_kots_trajectory_problem

    model = Kots.from_urdf_file(str(ROOT / "examples/models/planar2.urdf"))
    spec = problem()
    spec["model"] = {"file": "/not-present/robot.urdf", "locked_joints": {"joint2": .2}}
    compiled = compile_kots_trajectory_problem(spec, model=model)
    assert compiled.state_builder.model is model
    assert compiled.trajectory_map.q_dim == 2


@pytest.mark.parametrize("backend", ["kots", "pinocchio"])
def test_file_loading_and_locks_through_ioc(backend):
    pytest.importorskip("robokots.kots" if backend == "kots" else "pinocchio")
    from rei.optimize_backends.trajectory_ioc import compile_trajectory_ioc_problem

    spec = problem()
    spec["model"]["locked_joints"] = {"joint2": .2}
    compiled = compile_trajectory_ioc_problem(spec, backend=backend)
    assert compiled.trajectory_map.q_dim == 1
    r, J = compiled.runtime.linearize()
    assert np.isfinite(r).all() and np.isfinite(J).all()


@pytest.mark.parametrize("config,error", [
    ({}, ValueError),
    ({"file": "/not-present/robot.urdf"}, FileNotFoundError),
    ({"file": "robot.obj"}, ValueError),
    ({"file": str(ROOT / "examples/models/planar2.json"), "locked_joints": {"joint2": .2}}, ValueError),
])
def test_loading_errors_are_explicit(config, error):
    pytest.importorskip("robokots.kots")
    from rei.optimize_backends.kots import compile_kots_trajectory_problem

    spec = problem()
    spec["model"] = config
    with pytest.raises(error):
        compile_kots_trajectory_problem(spec)
