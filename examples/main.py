from __future__ import annotations

from pathlib import Path

import numpy as np

from eiopt import compile_problem, load_problem_toml, solve_gauss_newton

_DEFAULT_DSL_PATH = Path(__file__).parent / "specs" / "basic.toml"


def main() -> None:
    dsl = load_problem_toml(_DEFAULT_DSL_PATH)

    def build_state_backend(x_all: np.ndarray, *, required=None, **_kwargs) -> dict:
        q = np.asarray(x_all, dtype=float).reshape(-1)
        pos = np.array([float(q[0]), float(q[1]), 0.0], dtype=float)
        J_pos = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float)

        out: dict = {}
        if required is None:
            return out

        for key in required:
            if getattr(key, "dtype", None) == "frame" and getattr(key, "field", None) == "pos":
                out[key] = pos
            elif getattr(key, "dtype", None) == "frame" and getattr(key, "field", None) == "pos_J_q":
                out[key] = J_pos
        return out

    problem, ctx, required = compile_problem(dsl, build_state=build_state_backend)

    q0 = ctx.pack.vars[0].x.copy()
    x_star, _cost, _iters, _rnorm, _dxnorm, _converged = solve_gauss_newton(
        problem, ctx.pack, max_iters=10, ctx=ctx, required=required
    )

    print("q0:", q0)
    print("q*:", x_star)


if __name__ == "__main__":
    main()
