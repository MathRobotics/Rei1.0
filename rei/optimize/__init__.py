from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ProblemPoint": ("rei.problem", "ProblemPoint"),
    "EvaluateProblem": ("rei.problem", "EvaluateProblem"),
    "LinearizedProblem": ("rei.problem", "LinearizedProblem"),
    "OperatorProblem": ("rei.problem", "OperatorProblem"),
    "ProjectProblem": ("rei.problem", "ProjectProblem"),
    "ConstraintProblem": ("rei.problem", "ConstraintProblem"),
    "NLSRuntimeLinearProblem": ("rei.problem", "NLSRuntimeLinearProblem"),
    "NLSRuntimeConstraintProblem": ("rei.problem", "NLSRuntimeConstraintProblem"),
    "as_linearized_problem": ("rei.problem", "as_linearized_problem"),
    "NLSProblem": ("rei.problem", "NLSProblem"),
    "NLSRuntime": ("rei.optimize.runtime", "NLSRuntime"),
    "LinearizedTerm": ("rei.optimize.runtime", "LinearizedTerm"),
    "StackedTermSlice": ("rei.optimize.runtime", "StackedTermSlice"),
    "collect_required": ("rei.optimize.runtime", "collect_required"),
    "Cost": ("rei.optimize.costs", "Cost"),
    "L2Cost": ("rei.optimize.costs", "L2Cost"),
    "DiagonalWeightCost": ("rei.optimize.costs", "DiagonalWeightCost"),
    "ScalarWeightCost": ("rei.optimize.costs", "ScalarWeightCost"),
    "HuberCost": ("rei.optimize.costs", "HuberCost"),
    "DslBuildEnv": ("rei.optimize.dsl", "DslBuildEnv"),
    "iter_nodes": ("rei.optimize.dsl", "iter_nodes"),
    "rewrite_get_state_owner_name": ("rei.optimize.dsl", "rewrite_get_state_owner_name"),
    "find_const_expr": ("rei.optimize.dsl", "find_const_expr"),
    "find_var_dsl": ("rei.optimize.dsl", "find_var_dsl"),
    "split_terms_by_component": ("rei.optimize.dsl", "split_terms_by_component"),
    "build_trajectory_map": ("rei.optimize.dsl", "build_trajectory_map"),
    "build_trajectory_map_with_derivative": ("rei.optimize.dsl", "build_trajectory_map_with_derivative"),
    "build_trajectory_maps_with_derivatives": ("rei.optimize.dsl", "build_trajectory_maps_with_derivatives"),
    "default_dt_from_time": ("rei.optimize.dsl", "default_dt_from_time"),
    "default_steps_from_time": ("rei.optimize.dsl", "default_steps_from_time"),
    "infer_bspline_q_dim_from_var": ("rei.optimize.dsl", "infer_bspline_q_dim_from_var"),
    "PreparedTrajectoryProblemDsl": ("rei.optimize.dsl", "PreparedTrajectoryProblemDsl"),
    "prepare_trajectory_problem_dsl": ("rei.optimize.dsl", "prepare_trajectory_problem_dsl"),
    "resolve_variable_dim": ("rei.optimize.dsl", "resolve_variable_dim"),
    "expand_variable_init": ("rei.optimize.dsl", "expand_variable_init"),
    "normalize_variable_dsl": ("rei.optimize.dsl", "normalize_variable_dsl"),
    "NullspaceReducedRuntime": ("rei.optimize.reductions", "NullspaceReducedRuntime"),
    "NullspaceEqualityReduction": ("rei.optimize.reductions", "NullspaceEqualityReduction"),
    "build_nullspace_equality_reduction": ("rei.optimize.reductions", "build_nullspace_equality_reduction"),
    "scale_matrix_with_projection_svd": ("rei.optimize.reductions", "scale_matrix_with_projection_svd"),
    "register_default_costs": ("rei.optimize.builder", "register_default_costs"),
    "create_default_expr_register": ("rei.optimize.builder", "create_default_expr_register"),
    "build_variable": ("rei.optimize.builder", "build_variable"),
    "build_variable_pack": ("rei.optimize.builder", "build_variable_pack"),
    "build_term": ("rei.optimize.builder", "build_term"),
    "build_nls_problem": ("rei.optimize.builder", "build_nls_problem"),
    "collect_required_state_keys": ("rei.optimize.builder", "collect_required_state_keys"),
    "compile_nls_problem": ("rei.optimize.builder", "compile_nls_problem"),
    "load_problem_toml": ("rei.optimize.builder", "load_problem_toml"),
    "format_solve_report": ("rei.optimize.report", "format_solve_report"),
    "format_timing_report": ("rei.optimize.report", "format_timing_report"),
    "IterRow": ("rei.optimize.textlog", "IterRow"),
    "IterCallback": ("rei.optimize.textlog", "IterCallback"),
    "build_solver_iter_logger": ("rei.optimize.textlog", "build_solver_iter_logger"),
    "compress_iter_history": ("rei.optimize.textlog", "compress_iter_history"),
    "format_numeric_array": ("rei.optimize.textlog", "format_numeric_array"),
    "build_timestamped_log_path": ("rei.optimize.textlog", "build_timestamped_log_path"),
    "write_text_log": ("rei.optimize.textlog", "write_text_log"),
    "format_solver_text_log": ("rei.optimize.textlog", "format_solver_text_log"),
    "KKTCheckResult": ("rei.optimize.kkt", "KKTCheckResult"),
    "check_kkt_residuals": ("rei.optimize.kkt", "check_kkt_residuals"),
    "check_kkt_conditions": ("rei.optimize.kkt", "check_kkt_conditions"),
    "collect_named_expr_values": ("rei.optimize.report", "collect_named_expr_values"),
    "get_named_expr_value": ("rei.optimize.report", "get_named_expr_value"),
    "TermAttrPlotSeries": ("rei.optimize.plot", "TermAttrPlotSeries"),
    "collect_plot_series_from_term_attrs": ("rei.optimize.plot", "collect_plot_series_from_term_attrs"),
    "collect_trajectory_derivative_plot_series": ("rei.optimize.plot", "collect_trajectory_derivative_plot_series"),
    "plot_term_attrs": ("rei.optimize.plot", "plot_term_attrs"),
    "build_term_gradient_matrix": ("rei.optimize.term_gradient_matrix", "build_term_gradient_matrix"),
    "build_term_gradient_matrix_from_stacked": (
        "rei.optimize.term_gradient_matrix",
        "build_term_gradient_matrix_from_stacked",
    ),
    "build_term_gradient_matrix_from_terms": (
        "rei.optimize.term_gradient_matrix",
        "build_term_gradient_matrix_from_terms",
    ),
    "SolveStats": ("rei.core.outcome", "SolveStats"),
    "SolveOutcome": ("rei.core.outcome", "SolveOutcome"),
    "TimingSpan": ("rei.core.timing", "TimingSpan"),
    "TimingReport": ("rei.core.timing", "TimingReport"),
    "Profiler": ("rei.core.timing", "Profiler"),
    "nls": ("rei.optimize.solvers", "nls"),
    "solve": ("rei.optimize.solvers", "solve"),
    "solve_gauss_newton": ("rei.optimize.solvers", "solve_gauss_newton"),
    "solve_scipy_minimize": ("rei.optimize.solvers", "solve_scipy_minimize"),
    "solve_cyipopt_minimize": ("rei.optimize.solvers", "solve_cyipopt_minimize"),
    "solve_liteopt_gd": ("rei.optimize.solvers", "solve_liteopt_gd"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name, None)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = target
    mod = import_module(mod_name)
    value = getattr(mod, attr)
    globals()[name] = value
    return value
