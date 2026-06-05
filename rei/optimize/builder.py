from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..core.expr.types import Variable, VariablePack
from ..core.state_cache import StateCache, StateKey
from ..core.time_grid import TimeGrid
from .dsl.environment import DslBuildEnv
from .dsl.io import load_problem_toml
from .dsl.variable_utils import expand_variable_init, resolve_variable_dim
from ..core.expr.registry import ExprRegister
from ..problem import NLSProblem
from .runtime import NLSRuntime, collect_required as collect_required_from_problem
from .costs import DiagonalWeightCost, HuberCost, L2Cost, ScalarWeightCost

Array = np.ndarray


def _canonical_constraint_kind(kind: Any) -> str:
    value = str(kind).strip().lower()
    if value in ("eq", "equality"):
        return "eq"
    if value in ("ineq", "inequality"):
        return "ineq"
    raise ValueError(
        f"term constraint kind/type must be 'eq' or 'ineq'. Got {kind!r}."
    )


def _normalize_constraint_attrs(term_dsl: dict[str, Any], attrs: dict[str, Any]) -> dict[str, Any]:
    constraint_dsl = term_dsl.get("constraint", None)
    if constraint_dsl is not None:
        if isinstance(constraint_dsl, str):
            attrs["constraint_kind"] = _canonical_constraint_kind(constraint_dsl)
            attrs.setdefault("is_constraint", True)
        elif isinstance(constraint_dsl, dict):
            kind_raw = constraint_dsl.get("kind", constraint_dsl.get("type", None))
            if kind_raw is not None:
                attrs["constraint_kind"] = _canonical_constraint_kind(kind_raw)
                attrs.setdefault("is_constraint", True)

            is_constraint_raw = constraint_dsl.get("is_constraint", constraint_dsl.get("enabled", None))
            if is_constraint_raw is not None:
                attrs["is_constraint"] = bool(is_constraint_raw)
        else:
            raise ValueError("term.constraint must be a string or dict.")

    if "constraint_kind" in attrs:
        attrs["constraint_kind"] = _canonical_constraint_kind(attrs["constraint_kind"])
        attrs.setdefault("is_constraint", True)

    return attrs


def register_default_costs(expr_register: ExprRegister) -> None:
    expr_register.register_cost("l2", lambda dsl: L2Cost())
    expr_register.register_cost("diag_weight", lambda dsl: DiagonalWeightCost(w=np.asarray(dsl["w"], float)))
    expr_register.register_cost("scalar_weight", lambda dsl: ScalarWeightCost(w=float(dsl["w"])))
    expr_register.register_cost("huber", lambda dsl: HuberCost(delta=float(dsl["delta"])))


def create_default_expr_register() -> ExprRegister:
    from ..core.expr.stdlib import register_stdlib

    expr_register = ExprRegister()
    register_stdlib(expr_register)
    register_default_costs(expr_register)
    return expr_register


def build_variable(dsl: dict[str, Any]) -> Variable:
    name = str(dsl["name"])
    dim = resolve_variable_dim(dsl.get("dim", None), name=name)

    if "init" in dsl:
        x = expand_variable_init(dsl["init"], dim=dim, where=f"variable '{name}'")
        return Variable(name=name, x=x.copy())

    if dim is None:
        raise ValueError(f"variable '{name}': either dim or init is required.")
    return Variable(name=name, x=np.zeros((dim,), dtype=float))


def build_variable_pack(dsl: dict[str, Any]) -> VariablePack:
    return VariablePack([build_variable(v_dsl) for v_dsl in dsl.get("variables", [])])


def build_term(env: DslBuildEnv, dsl: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    expr_dsl = dsl.get("expr", None)
    if not isinstance(expr_dsl, dict):
        raise ValueError("term.expr must be a dict.")

    cost_dsl = dsl.get("cost", {"type": "l2"})
    if not isinstance(cost_dsl, dict):
        raise ValueError("term.cost must be a dict.")

    attrs_dsl = dsl.get("attrs", None)
    if attrs_dsl is None:
        attrs: dict[str, Any] = {}
    elif isinstance(attrs_dsl, dict):
        attrs = dict(attrs_dsl)
    else:
        raise ValueError("term.attrs must be a dict.")

    for key, value in dsl.items():
        if key in ("expr", "cost", "attrs", "constraint"):
            continue
        attrs[key] = value

    attrs = _normalize_constraint_attrs(dsl, attrs)

    expr = env.build_expr(expr_dsl)
    cost = env.build_cost(cost_dsl)
    return expr, cost, attrs


def build_nls_problem(dsl: dict[str, Any], *, expr_register: ExprRegister) -> tuple[NLSProblem, TimeGrid]:
    time_dsl = dsl.get("time", None)
    time = TimeGrid.single_time() if time_dsl is None else TimeGrid.from_dsl(time_dsl)

    pack = build_variable_pack(dsl)
    env = DslBuildEnv(pack=pack, time=time, expr_register=expr_register, root_dsl=dsl)
    built_terms = [build_term(env, term_dsl) for term_dsl in dsl.get("terms", [])]
    terms = [(expr, cost) for expr, cost, _attrs in built_terms]
    term_attrs = [attrs for _expr, _cost, attrs in built_terms]

    return NLSProblem(variables=pack, terms=terms, term_attrs=term_attrs), time


def collect_required_state_keys(problem: NLSProblem) -> list[StateKey]:
    return collect_required_from_problem(problem)


def compile_nls_problem(
    dsl: dict[str, Any],
    *,
    build_state: Callable[..., dict],
    expr_register: ExprRegister | None = None,
) -> NLSRuntime:
    if expr_register is None:
        expr_register = create_default_expr_register()

    problem, time = build_nls_problem(dsl, expr_register=expr_register)
    cache = StateCache(build_state=build_state)
    required = collect_required_state_keys(problem)
    return NLSRuntime.from_problem(problem, state=cache, time=time, required=required)


def compile_nls_problem_spec(
    spec: Mapping[str, Any],
    *,
    build_state: Callable[..., dict],
    expr_register: ExprRegister | None = None,
) -> NLSRuntime:
    from .dsl.spec import problem_spec_to_dsl

    return compile_nls_problem(
        problem_spec_to_dsl(spec),
        build_state=build_state,
        expr_register=expr_register,
    )


def compile_nls_problem_spec_toml(
    path: str | Path,
    *,
    build_state: Callable[..., dict],
    expr_register: ExprRegister | None = None,
) -> NLSRuntime:
    from .dsl.spec import load_problem_spec_toml

    return compile_nls_problem(
        load_problem_spec_toml(path),
        build_state=build_state,
        expr_register=expr_register,
    )


__all__ = [
    "NLSProblem",
    "NLSRuntime",
    "register_default_costs",
    "create_default_expr_register",
    "build_variable",
    "build_variable_pack",
    "build_term",
    "build_nls_problem",
    "collect_required_state_keys",
    "compile_nls_problem",
    "compile_nls_problem_spec",
    "compile_nls_problem_spec_toml",
    "load_problem_toml",
]
