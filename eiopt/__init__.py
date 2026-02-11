"""EiOpt: small optimization utilities.

This repository is a standalone extraction/re-implementation of the
`robokots.inward` modules so they can be used as an external library.
"""

from __future__ import annotations

from . import core, dsl, expr, model, solvers
from .solvers import nls, solve_cyipopt_minimize, solve_gauss_newton, solve_runtime, solve_scipy_minimize
from .dsl import compile_problem, load_problem_toml
from .model import ProblemRuntime
from .report import format_solve_report, collect_named_expr_values, get_named_expr_value

__all__ = [
    "core",
    "expr",
    "model",
    "solvers",
    "dsl",
    "nls",
    "solve_gauss_newton",
    "solve_scipy_minimize",
    "solve_cyipopt_minimize",
    "solve_runtime",
    "compile_problem",
    "load_problem_toml",
    "ProblemRuntime",
    "format_solve_report",
    "collect_named_expr_values",
    "get_named_expr_value",
]
