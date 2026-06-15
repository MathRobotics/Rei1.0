from __future__ import annotations

import numpy as np
import pytest

from rei.optimize.builder import compile_nls_problem
from rei import compile_nls_problem_spec, compile_nls_problem_spec_toml
from rei.optimize.dsl import load_problem_spec_toml, problem_spec_to_dsl


def test_problem_spec_to_dsl_builds_basic_quadratic() -> None:
    spec = {
        "variables": {
            "q": {"dim": 2, "init": [0.0, 0.0]},
        },
        "terms": [
            {
                "name": "q_target",
                "weight": 2.0,
                "residual": {"var": "q", "target": [0.5, -1.2]},
            },
        ],
    }

    dsl = problem_spec_to_dsl(spec)

    assert dsl["variables"] == [{"name": "q", "dim": 2, "init": [0.0, 0.0]}]
    assert dsl["terms"][0]["expr"] == {
        "type": "sub",
        "name": "q_target",
        "a": {"type": "get_var", "name": "q_target_value", "var": "q"},
        "b": {"type": "const", "name": "q_target_target", "var": "q", "value": [0.5, -1.2]},
    }
    assert dsl["terms"][0]["cost"] == {"type": "scalar_weight", "w": 2.0}

    runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
    r, J = runtime.linearize()
    assert np.allclose(r, np.sqrt(2.0) * np.array([-0.5, 1.2]))
    assert np.allclose(J, np.sqrt(2.0) * np.eye(2))

    runtime_spec = compile_nls_problem_spec(spec, build_state=lambda *_args, **_kwargs: {})
    r_spec, _J_spec = runtime_spec.linearize()
    assert np.allclose(r_spec, r)


def test_problem_spec_accepts_term_state_target_shorthand() -> None:
    spec = {
        "optimization_variables": {"q": {"dim": 2, "init": [0.0, 0.0]}},
        "terms": [
            {
                "name": "ee_pos",
                "state": "kinematics.link.ee.pos",
                "var": "q",
                "target": [0.4, 0.1, 0.3],
                "constraint": "eq",
            }
        ],
    }

    dsl = problem_spec_to_dsl(spec)

    assert dsl["terms"][0] == {
        "expr": {
            "type": "sub",
            "name": "ee_pos",
            "a": {
                "type": "get_state",
                "name": "ee_pos_value",
                "key": {
                    "k": 0,
                    "owner_type": "link",
                    "owner_name": "ee",
                    "dtype": "kinematics",
                    "field": "pos",
                },
                "jac": {"var": "q"},
            },
            "b": {
                "type": "const",
                "name": "ee_pos_target",
                "var": "q",
                "value": [0.4, 0.1, 0.3],
            },
        },
        "cost": {"type": "l2"},
        "constraint": {"kind": "eq"},
    }


def test_problem_spec_preserves_enforce_metadata() -> None:
    spec = {
        "variables": {"q": {"dim": 2, "init": [0.0, 0.0]}},
        "terms": [
            {
                "name": "q_init",
                "kind": "eq",
                "enforce": "nullspace",
                "var": "q",
                "target": [0.0, 0.0],
            }
        ],
    }

    dsl = problem_spec_to_dsl(spec)

    assert dsl["terms"][0]["constraint"] == {"kind": "eq"}
    assert dsl["terms"][0]["attrs"] == {"enforce": "nullspace"}


def test_problem_spec_preserves_plot_metadata() -> None:
    spec = {
        "variables": {"q": {"dim": 1, "init": [0.0]}},
        "terms": [
            {
                "name": "q_plot",
                "state": "coord.total_joint.robot.q",
                "var": "q",
                "target": [0.0],
                "plot": "joint_q",
            }
        ],
    }

    dsl = problem_spec_to_dsl(spec)

    assert dsl["terms"][0]["attrs"] == {"plot": "joint_q"}


def test_problem_spec_converts_quantity_plot_to_traj_derivative_metadata() -> None:
    spec = {
        "time": {"N": 3, "dt": 0.1},
        "trajectory": {"type": "bspline", "var": "p", "degree": 2, "num_ctrl_points": 3},
        "opt_vals": {"trajectory_params": {"init": {"fill": 0.0}}},
        "terms": [
            {
                "name": "q_init",
                "quantity": "joint_angles",
                "at": 0,
                "target": {"fill": 0.0},
                "plot": "joint_q",
            },
            {
                "name": "qdot_reg",
                "quantity": "joint_velocities",
                "plot": True,
            },
        ],
    }

    dsl = problem_spec_to_dsl(spec)

    assert dsl["terms"][0]["attrs"] == {
        "plot": {
            "type": "traj_derivative",
            "name": "joint_q",
            "derivative_order": 0,
        }
    }
    assert dsl["terms"][1]["attrs"] == {
        "plot": {
            "type": "traj_derivative",
            "derivative_order": 1,
        }
    }


def test_problem_spec_resolves_reserved_opt_vals() -> None:
    spec = {
        "opt_vals": {
            "joint_angles": {"dim": 2, "init": [0.0, 0.0]},
        },
        "terms": [
            {
                "name": "posture",
                "var": "joint_angles",
                "target": [0.2, -0.1],
            },
            {
                "name": "ee_pos",
                "state": "kinematics.link.ee.pos",
                "var": "joint_angles",
                "target": [0.4, 0.1, 0.3],
            },
        ],
    }

    dsl = problem_spec_to_dsl(spec)

    assert dsl["variables"] == [{"name": "q", "dim": 2, "init": [0.0, 0.0]}]
    assert dsl["terms"][0]["expr"]["a"] == {
        "type": "get_var",
        "name": "posture_value",
        "var": "q",
    }
    assert dsl["terms"][0]["expr"]["b"] == {
        "type": "const",
        "name": "posture_target",
        "var": "q",
        "value": [0.2, -0.1],
    }
    assert dsl["terms"][1]["expr"]["a"]["jac"] == {"var": "q"}
    assert dsl["terms"][1]["expr"]["b"]["var"] == "q"


def test_problem_spec_resolves_reserved_trajectory_quantities() -> None:
    spec = {
        "time": {"N": 3, "dt": 0.1},
        "trajectory": {"type": "bspline", "var": "p", "degree": 2, "num_ctrl_points": 3},
        "opt_vals": {"trajectory_params": {"init": {"fill": 0.0}}},
        "terms": [
            {
                "name": "qdot_init",
                "quantity": "joint_velocities",
                "at": 0,
                "target": {"fill": 0.0},
            },
            {
                "name": "qddot_reg",
                "quantity": "joint_accelerations",
            },
            {
                "name": "named_qdot",
                "residual": {"name": "custom_qdot_expr", "quantity": "joint_velocities"},
            },
            {
                "name": "q_upper",
                "residual": {
                    "op": "hinge",
                    "base": {
                        "op": "sub",
                        "a": {"quantity": "joint_angles"},
                        "b": {"const_repeat": {"fill": 3.14}, "var": "trajectory_params"},
                    },
                },
            },
        ],
    }

    dsl = problem_spec_to_dsl(spec)

    assert dsl["terms"][0]["expr"] == {
        "type": "sub",
        "name": "qdot_init",
        "a": {
            "type": "get_traj_var",
            "name": "qdot_init_value",
            "var": "p",
            "k": 0,
            "derivative_order": 1,
            "derivative_wrt": "time",
        },
        "b": {
            "type": "const",
            "name": "qdot_init_target",
            "var": "p",
            "value": {"fill": 0.0},
        },
    }
    assert dsl["terms"][1]["expr"] == {
        "type": "get_traj_var",
        "name": "qddot_reg",
        "var": "p",
        "derivative_order": 2,
        "derivative_wrt": "time",
    }
    assert dsl["terms"][2]["expr"] == {
        "type": "get_traj_var",
        "name": "custom_qdot_expr",
        "var": "p",
        "derivative_order": 1,
        "derivative_wrt": "time",
    }
    assert dsl["terms"][3]["expr"]["base"]["a"] == {
        "type": "get_traj_var",
        "name": "q_upper_base_a",
        "var": "p",
        "derivative_order": 0,
    }


def test_problem_spec_converts_trajectory_quantity_bounds() -> None:
    spec = {
        "time": {"N": 3, "dt": 0.1},
        "trajectory": {"type": "bspline", "var": "p", "degree": 2, "num_ctrl_points": 3},
        "opt_vals": {"trajectory_params": {"init": {"fill": 0.0}}},
        "terms": [
            {
                "name": "joint_q_bounds",
                "kind": "ineq",
                "quantity": "joint_angles",
                "bounds": {
                    "lower": {"fill": -3.14},
                    "upper": {"fill": 3.14},
                },
                "weight": 1e5,
            },
        ],
    }

    dsl = problem_spec_to_dsl(spec)
    expr = dsl["terms"][0]["expr"]

    assert dsl["terms"][0]["constraint"] == {"kind": "ineq"}
    assert expr["type"] == "vstack"
    assert expr["name"] == "joint_q_bounds"
    assert [part["name"] for part in expr["parts"]] == [
        "joint_q_bounds_upper_violation",
        "joint_q_bounds_lower_violation",
    ]
    assert expr["parts"][0]["base"] == {
        "type": "sub",
        "name": "joint_q_bounds_upper_margin",
        "a": {
            "type": "get_traj_var",
            "name": "joint_q_bounds_upper_value",
            "var": "p",
            "derivative_order": 0,
        },
        "b": {
            "type": "const_repeat",
            "name": "joint_q_bounds_upper",
            "value": {"fill": 3.14},
            "var": "p",
        },
    }
    assert expr["parts"][1]["base"] == {
        "type": "sub",
        "name": "joint_q_bounds_lower_margin",
        "a": {
            "type": "const_repeat",
            "name": "joint_q_bounds_lower",
            "value": {"fill": -3.14},
            "var": "p",
        },
        "b": {
            "type": "get_traj_var",
            "name": "joint_q_bounds_lower_value",
            "var": "p",
            "derivative_order": 0,
        },
    }


def test_problem_spec_converts_joint_torque_quantity() -> None:
    spec = {
        "time": {"N": 10, "dt": 0.1},
        "trajectory": {"type": "bspline", "var": "p", "degree": 2, "num_ctrl_points": 3},
        "opt_vals": {"trajectory_params": {"init": {"fill": 0.0}}},
        "terms": [
            {
                "name": "torque_traj_regularization",
                "quantity": "joint_torques",
                "stride": 5,
                "plot": {"name": "joint_torque", "stride": 5},
                "weight": 1e-10,
            },
        ],
    }

    dsl = problem_spec_to_dsl(spec)

    assert dsl["terms"][0]["attrs"] == {"plot": {"name": "joint_torque", "stride": 5}}
    assert dsl["terms"][0]["expr"] == {
        "type": "stack",
        "name": "torque_traj_regularization",
        "range": {"k0": 0, "k1": "last", "stride": 5},
        "inner": {
            "type": "get_state",
            "name": "torque_traj_regularization_k",
            "key": {
                "k": 0,
                "owner_type": "total_joint",
                "owner_name": "robot",
                "dtype": "dynamics",
                "field": "torque",
            },
            "jac": {"var": "p"},
        },
    }


def test_problem_spec_rejects_unknown_quantity() -> None:
    with pytest.raises(ValueError, match="unknown quantity"):
        problem_spec_to_dsl(
            {
                "opt_vals": {"trajectory_params": {"init": {"fill": 0.0}}},
                "terms": [{"quantity": "fluid_velocity"}],
            }
        )


def test_problem_spec_rejects_duplicate_opt_val_canonical_variable() -> None:
    with pytest.raises(ValueError, match="defined more than once"):
        problem_spec_to_dsl(
            {
                "optimization_variables": {"q": 2},
                "opt_vals": {"joint_angles": 2},
                "terms": [],
            }
        )


def test_problem_spec_rejects_mixed_optimization_variable_sections() -> None:
    with pytest.raises(ValueError, match="both optimization_variables and variables"):
        problem_spec_to_dsl(
            {
                "optimization_variables": {"q": 2},
                "variables": {"q": 2},
                "terms": [],
            }
        )


def test_problem_spec_toml_loader_is_standard_text_entrypoint(tmp_path) -> None:
    path = tmp_path / "problem.toml"
    path.write_text(
        """
[opt_vals.joint_angles]
dim = 2
init = [0.0, 0.0]

[[terms]]
name = "ee_pos"
state = "kinematics.link.ee.pos"
var = "joint_angles"
target = [0.4, 0.1, 0.3]
constraint = "eq"
""",
        encoding="utf-8",
    )

    dsl = load_problem_spec_toml(path)
    runtime = compile_nls_problem_spec_toml(path, build_state=lambda *_args, **_kwargs: {})

    assert runtime.pack.n_total == 2
    assert dsl["variables"] == [{"name": "q", "dim": 2, "init": [0.0, 0.0]}]
    assert dsl["terms"][0]["constraint"] == {"kind": "eq"}
    assert dsl["terms"][0]["expr"]["a"]["key"] == {
        "k": 0,
        "owner_type": "link",
        "owner_name": "ee",
        "dtype": "kinematics",
        "field": "pos",
    }


def test_problem_spec_converts_vision_and_stack_sections() -> None:
    dsl = problem_spec_to_dsl(
        {
            "time": {"N": 2, "dt": 0.1},
            "trajectory": {"type": "bspline", "var": "p", "degree": 2, "num_ctrl_points": 3},
            "vision": {"p_var": "theta", "owner_name": "cam0", "observations": [1.0, 2.0]},
            "variables": {"p": {"dim": 3, "init": {"fill": 0.0}}},
            "terms": [
                {
                    "name": "torque_stack",
                    "weight": 1e-4,
                    "residual": {
                        "op": "stack",
                        "range": {"k0": 0, "k1": "last"},
                        "inner": {
                            "state": "torque_k",
                            "var": "p",
                            "owner_type": "total_joint",
                            "owner": "robot",
                            "dtype": "dynamics",
                            "field": "torque",
                        },
                    },
                }
            ],
        }
    )

    assert dsl["vision"]["owner_name"] == "cam0"
    assert dsl["terms"][0]["expr"] == {
        "type": "stack",
        "name": "torque_stack",
        "range": {"k0": 0, "k1": "last"},
        "inner": {
            "type": "get_state",
            "name": "torque_k",
            "key": {
                "k": 0,
                "owner_type": "total_joint",
                "owner_name": "robot",
                "dtype": "dynamics",
                "field": "torque",
            },
            "jac": {"var": "p"},
        },
    }


def test_problem_spec_joint_target_uses_same_term_for_direct_joint_variable() -> None:
    spec = {
        "time": {"N": 2, "dt": 0.1},
        "joint": {"var": "q"},
        "variables": {"q": {"dim": 6, "init": {"fill": 0.0}}},
        "terms": [
            {
                "type": "joint_target",
                "name": "q_mid",
                "at": 1,
                "target": [1.0, -2.0],
            },
            {
                "type": "joint_target",
                "name": "q_final_fill",
                "at": "last",
                "target": {"fill": 0.5},
            },
        ],
    }

    dsl = problem_spec_to_dsl(spec)

    assert dsl["terms"][0]["expr"] == {
        "type": "sub",
        "name": "q_mid",
        "a": {"type": "get_var", "name": "q_mid_value", "var": "q", "k": 1},
        "b": {"type": "const", "name": "q_mid_target", "var": "q", "value": [1.0, -2.0], "dim": 2},
    }
    assert dsl["terms"][1]["expr"]["b"] == {
        "type": "const",
        "name": "q_final_fill_target",
        "var": "q",
        "value": {"fill": 0.5},
        "dim": 2,
    }

    runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
    r, J = runtime.linearize()
    assert np.allclose(r, np.array([-1.0, 2.0, -0.5, -0.5]))
    assert J.shape == (4, 6)
    assert np.allclose(J[:2, 2:4], np.eye(2))
    assert np.allclose(J[2:, 4:6], np.eye(2))


def test_problem_spec_joint_target_uses_same_term_for_bspline_trajectory() -> None:
    spec = {
        "time": {"N": 2, "dt": 0.1},
        "trajectory": {
            "type": "bspline",
            "var": "p",
            "q_dim": 2,
            "degree": 1,
            "num_ctrl_points": 2,
        },
        "variables": {"p": {"dim": 4, "init": {"fill": 0.0}}},
        "terms": [
            {
                "type": "joint_target",
                "name": "q_mid",
                "at": 1,
                "target": [1.0, -2.0],
            },
            {
                "type": "joint_target",
                "name": "q_final_fill",
                "at": "last",
                "target": {"fill": 0.5},
            },
        ],
    }

    dsl = problem_spec_to_dsl(spec)

    assert dsl["terms"][0]["expr"] == {
        "type": "sub",
        "name": "q_mid",
        "a": {"type": "get_traj_var", "name": "q_mid_value", "var": "p", "k": 1},
        "b": {"type": "const", "name": "q_mid_target", "var": "p", "value": [1.0, -2.0]},
    }
    assert dsl["terms"][1]["expr"]["a"] == {
        "type": "get_traj_var",
        "name": "q_final_fill_value",
        "var": "p",
        "k": "last",
    }

    runtime = compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})
    r, J = runtime.linearize()
    assert np.allclose(r, np.array([-1.0, 2.0, -0.5, -0.5]))
    assert J.shape == (4, 4)
    assert np.allclose(J[:2, :], np.array([[0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5]]))
    assert np.allclose(J[2:, :], np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))
