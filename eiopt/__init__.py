"""EiOpt package entrypoint.

The canonical API is now exposed under `eiopt.optimize` and
`eiopt.optimize_backends`.
"""

from __future__ import annotations

from . import core, optimize, optimize_backends
from .optimize import (
    NLSProblem,
    NLSRuntime,
    KKTCheckResult,
    build_term_gradient_matrix,
    build_term_gradient_matrix_from_stacked,
    build_term_gradient_matrix_from_terms,
    check_kkt_conditions,
    check_kkt_residuals,
    collect_named_expr_values,
    collect_plot_series_from_term_attrs,
    compile_nls_problem,
    estimate_weights_simplex,
    format_ioc_report,
    format_solve_report,
    get_named_expr_value,
    IocPreparationResult,
    IocTermInfo,
    load_problem_toml,
    nls,
    plot_term_attrs,
    prepare_ioc_weights,
    solve,
    solve_cyipopt_minimize,
    solve_gauss_newton,
    solve_liteopt_gd,
    solve_scipy_minimize,
    split_terms_by_component,
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
    "solve_liteopt_gd",
    "compile_nls_problem",
    "load_problem_toml",
    "NLSProblem",
    "NLSRuntime",
    "KKTCheckResult",
    "check_kkt_conditions",
    "check_kkt_residuals",
    "format_solve_report",
    "collect_named_expr_values",
    "get_named_expr_value",
    "collect_plot_series_from_term_attrs",
    "plot_term_attrs",
    "build_term_gradient_matrix",
    "build_term_gradient_matrix_from_stacked",
    "build_term_gradient_matrix_from_terms",
    "estimate_weights_simplex",
    "IocTermInfo",
    "IocPreparationResult",
    "prepare_ioc_weights",
    "format_ioc_report",
    "split_terms_by_component",
]
