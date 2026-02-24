"""Equation-oriented problem definitions (root, operator, dynamics, etc.)."""

from .problem import (
    EquationPoint,
    EvaluateEquation,
    LinearizedEquation,
    OperatorEquation,
    as_linear_equation_problem,
)
from .report import (
    build_ioc_log_sections,
    format_ioc_report,
)
from .simplex import (
    SimplexMinNormProblem,
    solve_projected_linearized_min_norm,
    solve_simplex_min_norm,
)
from .stationarity import (
    RuntimeStationaritySource,
    StationaritySource,
    StationarityTermContribution,
    build_reference_simplex_init,
    build_stationarity_gradient_matrix,
    filter_stationarity_contributions,
    normalize_simplex_nonnegative,
    select_active_stationarity_indices,
    term_constraint_kind,
)

__all__ = [
    "EquationPoint",
    "EvaluateEquation",
    "LinearizedEquation",
    "OperatorEquation",
    "as_linear_equation_problem",
    "build_ioc_log_sections",
    "format_ioc_report",
    "SimplexMinNormProblem",
    "solve_projected_linearized_min_norm",
    "solve_simplex_min_norm",
    "StationaritySource",
    "StationarityTermContribution",
    "RuntimeStationaritySource",
    "normalize_simplex_nonnegative",
    "term_constraint_kind",
    "filter_stationarity_contributions",
    "build_stationarity_gradient_matrix",
    "select_active_stationarity_indices",
    "build_reference_simplex_init",
]
