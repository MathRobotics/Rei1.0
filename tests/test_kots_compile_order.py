"""Automatic motion order must agree with explicitly configured RoboKots."""

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def problem(expr):
    return {
        "time": {"N": 3, "dt": 0.2},
        "trajectory": {"type": "bspline", "degree": 5, "num_ctrl_points": 6},
        "variables": [{"name": "p", "init": {"fill": 0.1}}],
        "terms": [{"expr": expr, "cost": {"type": "l2"}}],
    }


def state(field):
    return {
        "type": "get_state",
        "key": {
            "k": 1,
            "dtype": "dynamics",
            "owner_type": "total_joint",
            "owner_name": "robot",
            "field": field,
        },
    }


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("expr,order", [
    ({"type": "get_traj_var", "k": 1, "derivative_order": 0}, 1),
    ({"type": "get_traj_var", "k": 1, "derivative_order": 1}, 2),
    (state("torque"), 3),
    (state("torque_d1"), 4),
    (state("torque_d2"), 5),
    (state("torque_d3"), 6),
    (state("force"), 3),
    (state("momentum"), 2),
])
def test_compile_sets_order_and_matches_reference(backend, expr, order):
    Kots = pytest.importorskip("robokots.kots").Kots
    from rei.optimize_backends.kots import compile_kots_trajectory_problem

    path = str(ROOT / "examples/models/planar2.urdf")
    model = Kots.from_urdf_file(path, order=6 if order < 3 else 1, backend=backend)
    model.gravity_ = np.array([0.0, -9.81, 0.0])
    reference = Kots.from_urdf_file(path, order=order, backend=backend)
    reference.gravity_ = model.gravity_.copy()
    dsl = problem(expr)
    compiled = compile_kots_trajectory_problem(dsl, model=model, kots_backend=backend)
    expected = compile_kots_trajectory_problem(dsl, model=reference, kots_backend=backend)
    assert model.order() == compiled.model_order == order
    assert set(compiled.trajectory_derivative_maps) == set(range(order))
    np.testing.assert_array_equal(model.gravity_, reference.gravity_)
    point = np.random.default_rng(8).normal(scale=0.1, size=compiled.runtime.pack.n_total)
    compiled.runtime.pack.set(point)
    expected.runtime.pack.set(point)
    r, J = compiled.runtime.linearize()
    ref_r, ref_J = expected.runtime.linearize()
    np.testing.assert_allclose(r, ref_r)
    np.testing.assert_allclose(J, ref_J)


def test_explicit_options_and_same_order_do_not_reset_state():
    Kots = pytest.importorskip("robokots.kots").Kots
    from rei.optimize_backends.kots import compile_kots_trajectory_problem

    model = Kots.from_urdf_file(str(ROOT / "examples/models/planar2.urdf"), order=1)
    dsl = problem({"type": "get_traj_var", "k": 1})
    compiled = compile_kots_trajectory_problem(
        dsl, model=model, dynamics_fields=["torque_d1"], max_derivative_order=4,
    )
    assert compiled.model_order == 5
    motions = model.motions_
    compile_kots_trajectory_problem(dsl, model=model, max_derivative_order=4)
    assert model.motions_ is motions


def test_skipped_terms_do_not_increase_model_order():
    Kots = pytest.importorskip("robokots.kots").Kots
    from rei.optimize_backends.kots import compile_kots_trajectory_problem

    model = Kots.from_urdf_file(str(ROOT / "examples/models/planar2.urdf"), order=5)
    dsl = problem({"type": "get_traj_var", "k": 1})
    unsupported = state("torque_d3")
    unsupported["key"]["owner_type"] = "link"
    dsl["terms"].append({"expr": unsupported, "cost": {"type": "l2"}})
    compiled = compile_kots_trajectory_problem(dsl, model=model, unsupported="warn_skip")
    assert compiled.model_order == 1
    assert len(compiled.diagnostics.unsupported_terms) == 1
