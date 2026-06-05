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
