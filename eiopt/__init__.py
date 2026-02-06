"""EiOpt: small optimization utilities.

This repository is a standalone extraction/re-implementation of the
`robokots.inward` modules so they can be used as an external library.
"""

from __future__ import annotations

from . import adapters, core, expr, model, solvers, spec
from .adapters import with_standard_joint_q
from .solvers import nls, solve_gauss_newton
from .spec import build_problem_from_spec

__all__ = [
    "adapters",
    "core",
    "expr",
    "model",
    "solvers",
    "spec",
    "with_standard_joint_q",
    "nls",
    "solve_gauss_newton",
    "build_problem_from_spec",
]
