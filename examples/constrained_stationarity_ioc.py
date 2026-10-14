"""Recover weights from a convex demonstration with one active inequality."""
from __future__ import annotations

import numpy as np

from rei import compile_nls_problem_spec
from rei.equations import KktConstraintData, RuntimeStationaritySource, estimate_kkt_weights


def main() -> None:
    targets = np.array([[3., 0.], [3., 2.]])
    truth = np.array([.75, .25])
    # The weighted target mean is the unconstrained quadratic minimum. Project
    # onto x<=0 to generate the demonstration without an external solver.
    demonstration = truth @ targets
    demonstration[0] = min(demonstration[0], 0.)
    runtime = compile_nls_problem_spec({
        "opt_vals": {"x": {"init": demonstration.tolist()}},
        "terms": [
            {"name": f"target_{i}", "var": "x", "target": target.tolist()}
            for i, target in enumerate(targets)
        ] + [{"name": "cap", "var": "x", "bounds": {"upper": [0., 10.]}, "kind": "ineq"}],
    }, build_state=lambda *args, **kwargs: {})
    h, Jh = runtime.linearize_stacked_terms(
        weighted=False, term_indices=runtime.find_constraint_term_indices(kind="eq"),
    )
    g, Jg = runtime.linearize_inequality_constraints()
    result = estimate_kkt_weights(RuntimeStationaritySource(runtime), KktConstraintData(h, Jh, g, Jg))
    recovered = np.asarray(result["weights"]) @ targets
    recovered[0] = min(recovered[0], 0.)
    print("demonstration:", demonstration)
    print("known weights:", truth)
    print("estimated weights:", result["weights"])
    print("inequality multipliers:", result["kkt"]["inequality_multipliers"])
    print("KKT satisfied:", result["kkt"]["ok"])
    print("unique weights:", result["identifiability"]["unique_stationary_weights"])
    print("stationarity norm:", result["stationarity"]["ikkt_residual_norm"])
    print("reconstruction error:", np.linalg.norm(recovered - demonstration))


if __name__ == "__main__":
    main()
