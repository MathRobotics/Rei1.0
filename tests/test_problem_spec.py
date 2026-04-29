from __future__ import annotations

import json

import numpy as np

from rei.optimize.builder import compile_nls_problem
from rei import compile_nls_problem_spec, compile_nls_problem_spec_json
from rei.optimize.dsl import load_problem_spec_json, problem_spec_to_dsl


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


def test_problem_spec_json_loader_converts_constraint_state_target(tmp_path) -> None:
    path = tmp_path / "problem.json"
    path.write_text(
        json.dumps(
            {
                "time": {"N": 3, "dt": 0.1},
                "variables": {"p": {"dim": 4, "init": {"fill": 0.0}}},
                "terms": [
                    {
                        "name": "q_init",
                        "kind": "eq",
                        "weight": 100.0,
                        "attrs": {"nullspace_eq": True},
                        "residual": {
                            "state": "joint_q_init",
                            "var": "p",
                            "at": 0,
                            "owner_type": "total_joint",
                            "owner": "robot",
                            "dtype": "coord",
                            "field": "q",
                            "target": {"fill": 1.57},
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    dsl = load_problem_spec_json(path)
    runtime = compile_nls_problem_spec_json(path, build_state=lambda *_args, **_kwargs: {})
    assert runtime.pack.n_total == 4
    term = dsl["terms"][0]

    assert term["constraint"] == {"kind": "eq"}
    assert term["attrs"] == {"nullspace_eq": True}
    assert term["expr"]["a"] == {
        "type": "get_state",
        "name": "joint_q_init",
        "key": {
            "k": 0,
            "owner_type": "total_joint",
            "owner_name": "robot",
            "dtype": "coord",
            "field": "q",
        },
        "jac": {"var": "p"},
    }
    assert term["expr"]["b"] == {
        "type": "const",
        "name": "q_init_target",
        "var": "p",
        "value": {"fill": 1.57},
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
