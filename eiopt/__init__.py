"""EiOpt: small optimization utilities.

This repository is a standalone extraction/re-implementation of the
`robokots.inward` modules so they can be used as an external library.
"""

from __future__ import annotations

from . import core, dsl, expr, model, solvers
from .solvers import nls, solve_gauss_newton
from .dsl import compile_problem, load_problem_toml

__all__ = [
    "core",
    "expr",
    "model",
    "solvers",
    "dsl",
    "nls",
    "solve_gauss_newton",
    "compile_problem",
    "load_problem_toml",
]
