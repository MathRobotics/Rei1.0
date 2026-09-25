"""Fixed model uncertainty through real RoboKots and Rei derivative paths."""
from pathlib import Path
import builtins

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api():
    perturbation = pytest.importorskip("robokots.perturbation")
    from robokots.kots import Kots
    return Kots, perturbation


def model(api, backend="numpy"):
    return api[0].from_urdf_file(
        str(ROOT / "examples/models/planar2.urdf"), order=3, backend=backend,
    )


def problem():
    from rei import load_problem_spec_toml
    spec = load_problem_spec_toml(ROOT / "examples/spec/robokots_traj_dynamics_d12.toml")
    spec["time"].update(N=6, dt=.2)
    spec["trajectory"]["num_ctrl_points"] = 6
    return spec


@pytest.mark.parametrize("form", ["mapping", "spec", "path"])
def test_configuration_reproducible_and_nominal_unchanged(api, tmp_path, form):
    from rei.backends.state.robotics.kots import KotsStateBuilder
    nominal = model(api)
    before = nominal.robot_.to_dict()
    config = {"seed": 42, "mass_relative_std": .1, "link_translation_std": .002}
    value = config
    if form == "spec":
        value = api[1].PerturbationSpec.from_dict(config)
    elif form == "path":
        value = tmp_path / "noise.toml"
        value.write_text("seed = 42\nmass_relative_std = 0.1\nlink_translation_std = 0.002\n")
    first = KotsStateBuilder(nominal, perturbation=value)
    second = KotsStateBuilder(nominal, perturbation=value)
    assert first.model is not nominal
    assert nominal.robot_.to_dict() == before
    assert first.model.robot_.to_dict() == second.model.robot_.to_dict()
    assert first.model.robot_.to_dict() != before
    assert first.perturbation_report.to_dict() == second.perturbation_report.to_dict()
    assert config == {"seed": 42, "mass_relative_std": .1, "link_translation_std": .002}


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("batch", [False, True])
def test_perturbed_trajectory_values_and_products_match_reference(api, backend, batch):
    from rei.optimize_backends.kots import compile_kots_trajectory_problem
    nominal = model(api, backend)
    config = api[1].PerturbationSpec(seed=13, mass_relative_std=.1, link_translation_std=.002)
    reference_model = api[1].apply_perturbation(nominal, config)
    options = dict(kots_backend=backend, batch_trajectory=batch, gravity=(0., -9.81, 0.))
    compiled = compile_kots_trajectory_problem(problem(), model=nominal, perturbation=config, **options)
    reference = compile_kots_trajectory_problem(problem(), model=reference_model, **options)
    clean = compile_kots_trajectory_problem(problem(), model=nominal, **options)
    assert compiled.perturbation_report is compiled.state_builder.perturbation_report
    runtime = compiled.runtime
    rng = np.random.default_rng(12)
    point = rng.normal(scale=.1, size=runtime.pack.n_total)
    for target in (runtime, reference.runtime, clean.runtime):
        target.pack.set(point)
    r, J = runtime.linearize()
    ref_r, ref_J = reference.runtime.linearize()
    np.testing.assert_allclose(r, ref_r, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(J, ref_J, rtol=1e-11, atol=1e-11)
    assert np.linalg.norm(r - clean.runtime.linearize()[0]) > 1e-9
    v, w = rng.normal(size=J.shape[1]), rng.normal(size=J.shape[0])
    np.testing.assert_allclose(runtime.residual_jvp(v, weighted=True), J @ v, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(runtime.residual_vjp(w, weighted=True), J.T @ w, rtol=1e-9, atol=1e-8)
    eps = 1e-6
    runtime.pack.set(point + eps * v)
    plus = runtime.eval_stacked_terms(weighted=True)
    runtime.pack.set(point - eps * v)
    minus = runtime.eval_stacked_terms(weighted=True)
    np.testing.assert_allclose((plus-minus)/(2*eps), J @ v, rtol=1e-6, atol=1e-6)
    runtime.pack.set(point)
    np.testing.assert_allclose(runtime.eval_stacked_terms(weighted=True), r, rtol=0, atol=1e-12)


def test_template_retains_sample_and_ioc_forwards_configuration(api):
    from rei.optimize_backends.kots import compile_kots_trajectory_problem_template
    from rei.optimize_backends.trajectory_ioc import compile_trajectory_ioc_problem, estimate_ioc_weights
    config = {"seed": 4, "mass_relative_std": .05}
    nominal = model(api)
    template = compile_kots_trajectory_problem_template(problem(), model=nominal, perturbation=config)
    sampled_model = template.model
    report = template.compiled.perturbation_report.to_dict()
    point = np.linspace(-.1, .1, template.runtime.pack.n_total)
    template.update_window(p=point)
    assert template.model is sampled_model
    assert template.compiled.perturbation_report.to_dict() == report
    ioc = compile_trajectory_ioc_problem(problem(), backend="kots", model=nominal, data=None,
                                         perturbation=config)
    assert ioc.compiled.perturbation_report.to_dict() == report
    actual = estimate_ioc_weights(template)
    expected = estimate_ioc_weights(ioc, p=point)
    np.testing.assert_allclose(actual["weights"], expected["weights"])


def test_disabled_does_not_require_new_api_and_enabled_fails_clearly(api, monkeypatch):
    from rei.backends.state.robotics.kots import KotsStateBuilder
    nominal = model(api)
    real_import = builtins.__import__
    def without_perturbation(name, *args, **kwargs):
        if name == "robokots.perturbation":
            raise ImportError("old RoboKots")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_perturbation)
    builder = KotsStateBuilder(nominal)
    assert builder.model is nominal
    assert builder.perturbation_report is None
    with pytest.raises(ImportError, match="Update RoboKots"):
        KotsStateBuilder(nominal, perturbation={"seed": 1})


@pytest.mark.parametrize("config", [{"mass_relative_std": -1}, {"typo": .1}, 42])
def test_bad_configuration_is_rejected(api, config):
    from rei.backends.state.robotics.kots import KotsStateBuilder
    nominal = model(api)
    before = nominal.robot_.to_dict()
    with pytest.raises((TypeError, ValueError)):
        KotsStateBuilder(nominal, perturbation=config)
    assert nominal.robot_.to_dict() == before
