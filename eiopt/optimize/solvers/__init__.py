from __future__ import annotations

from .dispatch import (
    nls,
    solve,
    solve_cyipopt_minimize,
    solve_gauss_newton,
    solve_scipy_minimize,
)

__all__ = [
    "nls",
    "solve",
    "solve_gauss_newton",
    "solve_scipy_minimize",
    "solve_cyipopt_minimize",
]
