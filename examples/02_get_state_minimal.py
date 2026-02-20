from __future__ import annotations

import numpy as np

from rei.core.state_cache import OwnerKey, StateKey
from rei.core.state_schema import DTYPE_KINEMATICS, jac_field
from rei.optimize.builder import compile_nls_problem
from rei.optimize.report import format_solve_report
from rei.optimize.solvers import solve


def build_state(x_all, *, pack=None, time=None, required=None):
    del time

    q = np.asarray(x_all, dtype=float).reshape(-1)
    if pack is not None and hasattr(pack, "slices") and "q" in pack.slices:
        s, e = pack.slices["q"]
        q = np.asarray(q[s:e], dtype=float).reshape(-1)

    pos = np.array([q[0], q[1], 0.0], dtype=float)
    pos_jac = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    owner = OwnerKey(owner_type="link", owner_name="ee")
    key_pos = StateKey(
        k=0,
        owner=owner,
        dtype=DTYPE_KINEMATICS,
        field="pos",
        frame="world",
    )
    key_pos_jac = StateKey(
        k=0,
        owner=owner,
        dtype=DTYPE_KINEMATICS,
        field=jac_field("pos", var="q"),
        frame="world",
    )

    all_state = {
        key_pos: pos,
        key_pos_jac: pos_jac,
    }

    if required is None:
        return all_state

    req = set(required)
    return {k: v for k, v in all_state.items() if k in req}


def main() -> None:
    dsl = {
        "variables": [
            {"name": "q", "dim": 2, "init": [0.0, 0.0]},
        ],
        "terms": [
            {
                "expr": {
                    "type": "sub",
                    "name": "ee_pos_error",
                    "a": {
                        "type": "get_state",
                        "key": {
                            "k": 0,
                            "owner_type": "link",
                            "owner_name": "ee",
                            "dtype": DTYPE_KINEMATICS,
                            "field": "pos",
                        },
                        "jac": {"var": "q"},
                    },
                    "b": {"type": "const", "var": "q", "value": [1.5, -0.5, 0.0]},
                },
                "cost": {"type": "l2"},
            }
        ],
    }

    runtime = compile_nls_problem(dsl, build_state=build_state)

    x0 = runtime.pack.get().copy()
    out = solve(
        runtime,
        solver="gauss_newton",
        options={"max_iters": 50},
    )
    x_star = out.solution
    stats = out.stats

    print("=== 02_get_state_minimal ===")
    print(
        f"status={stats.status} converged={stats.converged} iters={stats.iterations} "
        f"cost0={float(stats.initial_objective or 0.0):.3e} "
        f"cost={float(stats.objective or 0.0):.3e} "
        f"rnorm={float(stats.residual_norm or 0.0):.3e} "
        f"dxnorm={float(stats.step_norm or 0.0):.3e}"
    )
    print(f"x0={x0}")
    print(f"x*={x_star}")
    print(format_solve_report(runtime, x0=x0, outcome=out))


if __name__ == "__main__":
    main()
