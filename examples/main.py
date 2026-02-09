from __future__ import annotations

from pathlib import Path

import numpy as np

from eiopt import compile_problem, load_problem_toml, solve_gauss_newton, with_standard_joint_q
from eiopt.core.state_schema import jac_field

_DEFAULT_DSL_PATH = Path(__file__).parent / "specs" / "basic.toml"


def main() -> None:
    """Minimal example: connect a backend via build_state() and solve an NLS problem."""

    # DSL: minimize || pos(q) - target ||^2 + 1e-6 * || q - q_nom ||^2
    # - `pos(q)` is provided by the backend via StateCache/build_state.
    # - `q/q_J_q` are injected automatically by `with_standard_joint_q`.
    dsl = load_problem_toml(_DEFAULT_DSL_PATH)

    def build_state_backend(
        x_all: np.ndarray,
        *,
        pack=None,
        time=None,
        required=None,
    ) -> dict:
        del time

        q = np.asarray(x_all, dtype=float).reshape(-1)
        if pack is not None and hasattr(pack, "slices") and "q" in pack.slices:
            s, e = pack.slices["q"]
            q = q[s:e]

        pos = np.array([float(q[0]), float(q[1]), 0.0], dtype=float)
        J_pos = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float)

        out: dict = {}
        if required is None:
            return out

        for key in required:
            if getattr(key, "dtype", None) != "frame":
                continue
            if getattr(key, "field", None) == "pos":
                out[key] = pos
            elif getattr(key, "field", None) == jac_field("pos", var="q"):
                out[key] = J_pos
        return out

    build_state = with_standard_joint_q(build_state_backend)
    problem, ctx, required = compile_problem(dsl, build_state=build_state)

    print("Required StateKeys:")
    for k in required:
        print(" ", k)

    q0 = ctx.pack.vars[0].x.copy()
    solve_gauss_newton(problem, ctx.pack, max_iters=10, ctx=ctx, required=required)
    q_star = ctx.pack.vars[0].x.copy()

    print("q0:", q0)
    print("q*:", q_star)


if __name__ == "__main__":
    main()
