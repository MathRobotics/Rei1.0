"""EiOpt package entrypoint.

The canonical API is now exposed under `eiopt.optimize` and
`eiopt.optimize_backends`.
"""

from __future__ import annotations

from . import core, optimize, optimize_backends
from .optimize import (
    NLSProblem,
    NLSRuntime,
    build_term_gradient_matrix,
    build_term_gradient_matrix_from_stacked,
    build_term_gradient_matrix_from_terms,
    collect_named_expr_values,
    compile_nls_problem,
    estimate_weights_simplex,
    format_solve_report,
    get_named_expr_value,
    load_problem_toml,
    nls,
    solve,
    solve_cyipopt_minimize,
    solve_gauss_newton,
    solve_scipy_minimize,
)

__all__ = [
    "core",
    "optimize",
    "optimize_backends",
    "nls",
    "solve",
    "solve_gauss_newton",
    "solve_scipy_minimize",
    "solve_cyipopt_minimize",
    "compile_nls_problem",
    "load_problem_toml",
    "NLSProblem",
    "NLSRuntime",
    "format_solve_report",
    "collect_named_expr_values",
    "get_named_expr_value",
    "build_term_gradient_matrix",
    "build_term_gradient_matrix_from_stacked",
    "build_term_gradient_matrix_from_terms",
    "estimate_weights_simplex",
]
