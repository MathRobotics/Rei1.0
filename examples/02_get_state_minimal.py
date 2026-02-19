from __future__ import annotations

import numpy as np

from eiopt.core.state_cache import OwnerKey, StateKey
from eiopt.core.state_schema import DTYPE_KINEMATICS, jac_field
from eiopt.optimize.builder import compile_nls_problem
from eiopt.optimize.report import format_solve_report
from eiopt.optimize.solvers import solve


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
    x_star, initial_cost, cost, iters, rnorm, dxnorm, converged = solve(
        runtime,
        solver="gauss_newton",
        options={"max_iters": 50},
    )

    print("=== 02_get_state_minimal ===")
    print(
        f"converged={converged} iters={iters} "
        f"cost0={initial_cost:.3e} cost={cost:.3e} "
        f"rnorm={rnorm:.3e} dxnorm={dxnorm:.3e}"
    )
    print(f"x0={x0}")
    print(f"x*={x_star}")
    print(format_solve_report(runtime, x0=x0, x_star=x_star))


if __name__ == "__main__":
    main()
