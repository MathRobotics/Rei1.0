from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "NLSProblem": ("eiopt.optimize.problem", "NLSProblem"),
    "NLSRuntime": ("eiopt.optimize.runtime", "NLSRuntime"),
    "LinearizedTerm": ("eiopt.optimize.runtime", "LinearizedTerm"),
    "StackedTermSlice": ("eiopt.optimize.runtime", "StackedTermSlice"),
    "collect_required": ("eiopt.optimize.runtime", "collect_required"),
    "Cost": ("eiopt.optimize.costs", "Cost"),
    "L2Cost": ("eiopt.optimize.costs", "L2Cost"),
    "DiagonalWeightCost": ("eiopt.optimize.costs", "DiagonalWeightCost"),
    "ScalarWeightCost": ("eiopt.optimize.costs", "ScalarWeightCost"),
    "HuberCost": ("eiopt.optimize.costs", "HuberCost"),
    "DslBuildEnv": ("eiopt.optimize.dsl", "DslBuildEnv"),
    "iter_nodes": ("eiopt.optimize.dsl", "iter_nodes"),
    "rewrite_get_state_owner_name": ("eiopt.optimize.dsl", "rewrite_get_state_owner_name"),
    "find_const_expr": ("eiopt.optimize.dsl", "find_const_expr"),
    "find_var_dsl": ("eiopt.optimize.dsl", "find_var_dsl"),
    "build_trajectory_map": ("eiopt.optimize.dsl", "build_trajectory_map"),
    "build_trajectory_map_with_derivative": ("eiopt.optimize.dsl", "build_trajectory_map_with_derivative"),
    "build_trajectory_maps_with_derivatives": ("eiopt.optimize.dsl", "build_trajectory_maps_with_derivatives"),
    "default_dt_from_time": ("eiopt.optimize.dsl", "default_dt_from_time"),
    "default_steps_from_time": ("eiopt.optimize.dsl", "default_steps_from_time"),
    "infer_bspline_q_dim_from_var": ("eiopt.optimize.dsl", "infer_bspline_q_dim_from_var"),
    "PreparedTrajectoryProblemDsl": ("eiopt.optimize.dsl", "PreparedTrajectoryProblemDsl"),
    "prepare_trajectory_problem_dsl": ("eiopt.optimize.dsl", "prepare_trajectory_problem_dsl"),
    "resolve_variable_dim": ("eiopt.optimize.dsl", "resolve_variable_dim"),
    "expand_variable_init": ("eiopt.optimize.dsl", "expand_variable_init"),
    "normalize_variable_dsl": ("eiopt.optimize.dsl", "normalize_variable_dsl"),
    "NullspaceReducedRuntime": ("eiopt.optimize.reductions", "NullspaceReducedRuntime"),
    "NullspaceEqualityReduction": ("eiopt.optimize.reductions", "NullspaceEqualityReduction"),
    "build_nullspace_equality_reduction": ("eiopt.optimize.reductions", "build_nullspace_equality_reduction"),
    "scale_matrix_with_projection_svd": ("eiopt.optimize.reductions", "scale_matrix_with_projection_svd"),
    "register_default_costs": ("eiopt.optimize.builder", "register_default_costs"),
    "create_default_expr_register": ("eiopt.optimize.builder", "create_default_expr_register"),
    "build_variable": ("eiopt.optimize.builder", "build_variable"),
    "build_variable_pack": ("eiopt.optimize.builder", "build_variable_pack"),
    "build_term": ("eiopt.optimize.builder", "build_term"),
    "build_nls_problem": ("eiopt.optimize.builder", "build_nls_problem"),
    "collect_required_state_keys": ("eiopt.optimize.builder", "collect_required_state_keys"),
    "compile_nls_problem": ("eiopt.optimize.builder", "compile_nls_problem"),
    "load_problem_toml": ("eiopt.optimize.builder", "load_problem_toml"),
    "format_solve_report": ("eiopt.optimize.report", "format_solve_report"),
    "collect_named_expr_values": ("eiopt.optimize.report", "collect_named_expr_values"),
    "get_named_expr_value": ("eiopt.optimize.report", "get_named_expr_value"),
    "build_term_gradient_matrix": ("eiopt.optimize.term_gradient_matrix", "build_term_gradient_matrix"),
    "build_term_gradient_matrix_from_stacked": (
        "eiopt.optimize.term_gradient_matrix",
        "build_term_gradient_matrix_from_stacked",
    ),
    "build_term_gradient_matrix_from_terms": (
        "eiopt.optimize.term_gradient_matrix",
        "build_term_gradient_matrix_from_terms",
    ),
    "estimate_weights_simplex": ("eiopt.optimize.simplex_weight_solver", "estimate_weights_simplex"),
    "nls": ("eiopt.optimize.solvers", "nls"),
    "solve": ("eiopt.optimize.solvers", "solve"),
    "solve_gauss_newton": ("eiopt.optimize.solvers", "solve_gauss_newton"),
    "solve_scipy_minimize": ("eiopt.optimize.solvers", "solve_scipy_minimize"),
    "solve_cyipopt_minimize": ("eiopt.optimize.solvers", "solve_cyipopt_minimize"),
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
