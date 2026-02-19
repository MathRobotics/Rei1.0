from __future__ import annotations

import argparse

import numpy as np

from eiopt import (
    RuntimeStationaritySource,
    build_reference_simplex_init,
    build_stationarity_gradient_matrix,
    compile_nls_problem,
    filter_stationarity_contributions,
    select_active_stationarity_indices,
    solve_simplex_min_norm,
    term_constraint_kind,
)


def build_demo_runtime(*, x_demo: float) -> object:
    dsl = {
        "variables": [{"name": "x", "dim": 1, "init": [float(x_demo)]}],
        "terms": [
            {
                "expr": {
                    "type": "sub",
                    "name": "x_to_1",
                    "a": {"type": "get_var", "var": "x"},
                    "b": {"type": "const", "var": "x", "value": [1.0]},
                },
                "cost": {"type": "scalar_weight", "w": 1.0},
            },
            {
                "expr": {
                    "type": "sub",
                    "name": "x_to_minus_1",
                    "a": {"type": "get_var", "var": "x"},
                    "b": {"type": "const", "var": "x", "value": [-1.0]},
                },
                "cost": {"type": "scalar_weight", "w": 1.0},
            },
            {
                "expr": {"type": "get_var", "name": "x_keep", "var": "x"},
                "cost": {"type": "scalar_weight", "w": 0.2},
            },
        ],
    }
    return compile_nls_problem(dsl, build_state=lambda *_args, **_kwargs: {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IOC-like stationarity example with simplex min-norm coefficients."
    )
    parser.add_argument(
        "--x-demo",
        type=float,
        default=0.0,
        help="Demonstration point x* used for stationarity matching (default: 0.0).",
    )
    parser.add_argument(
        "--active-mode",
        choices=["residual", "gradient"],
        default="residual",
        help="Active-set mode for stationarity solving (default: residual).",
    )
    args = parser.parse_args()

    runtime = build_demo_runtime(x_demo=float(args.x_demo))
    x_demo = runtime.pack.get().copy()

    source = RuntimeStationaritySource(runtime)
    source.set_point(x_demo)
    required = source.required_list(None)
    contributions_all = source.term_contributions(required=required)
    contributions = filter_stationarity_contributions(
        contributions_all,
        include_constraints=True,
    )
    A_col, term_indices = build_stationarity_gradient_matrix(
        contributions,
        n_total=int(source.n_total),
    )
    active_idx, active_grad_idx, active_res_idx = select_active_stationarity_indices(
        contributions,
        mode=str(args.active_mode),
    )

    w_hat = np.zeros((len(contributions),), dtype=float)
    solve_info: dict[str, object] | None = None
    if len(active_idx) > 0:
        x0 = build_reference_simplex_init(contributions, active_idx)
        w_active, solve_info_raw = solve_simplex_min_norm(
            np.asarray(A_col[:, active_idx], dtype=float),
            x0=x0,
            return_info=True,
        )
        w_hat[np.asarray(active_idx, dtype=int)] = np.asarray(w_active, dtype=float).reshape(-1)
        solve_info = dict(solve_info_raw)

    print("=== 10_stationarity_ioc ===")
    print(f"x_demo={x_demo}")
    print(f"term_indices={list(term_indices)}")
    print(f"A shape={A_col.shape}")
    print(f"active local idx ({args.active_mode})={list(active_idx)}")
    print(f"active local idx (gradient)={list(active_grad_idx)}")
    print(f"active local idx (residual)={list(active_res_idx)}")
    print(f"inferred={w_hat}")
    x0_ref = build_reference_simplex_init(contributions, active_idx)
    if x0_ref is not None:
        print(f"reference normalized (active only)={x0_ref}")
    if solve_info is not None:
        print(
            "simplex solver: "
            f"converged={bool(solve_info.get('converged', False))} "
            f"iters={int(solve_info.get('iterations', -1))} "
            f"objective={float(solve_info.get('objective', float('nan'))):.3e}"
        )

    print()
    for i, term in enumerate(contributions):
        is_constraint, kind = term_constraint_kind(dict(term.attrs))
        kind_str = "" if kind is None else f", kind={kind}"
        ttype = "constraint" if is_constraint else "objective"
        grad_norm = float(np.linalg.norm(np.asarray(term.gradient, dtype=float).reshape(-1)))
        print(
            f"[stationarity] term[{term.term_index}] {term.name} ({ttype}{kind_str}) "
            f"local={i} "
            f"||J^T r||={grad_norm:.6e} "
            f"||r_w||={float(term.weighted_residual_norm or 0.0):.6e} "
            f"w_hat={float(w_hat[i]):.6e}"
        )


if __name__ == "__main__":
    main()
