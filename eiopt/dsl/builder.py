from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..core.state_cache import StateCache, StateKey
from ..core.time_grid import TimeGrid
from ..expr.register_stdlib import register_stdlib
from ..expr.expr_register import ExprRegister
from ..model.problem import Problem
from ..model.runtime import ProblemRuntime, collect_required as collect_required_from_problem
from ..model.term import (
    DiagonalWeightCost,
    HuberCost,
    L2Cost,
    ScalarWeightCost,
    Variable,
    VariablePack,
)
from .environment import DslBuildEnv

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

    if "constraint_type" in attrs and "constraint_kind" not in attrs:
        attrs["constraint_kind"] = attrs["constraint_type"]

    if "constraint_kind" in attrs:
        attrs["constraint_kind"] = _canonical_constraint_kind(attrs["constraint_kind"])
        attrs["constraint_type"] = attrs["constraint_kind"]
        attrs.setdefault("is_constraint", True)

    return attrs


def register_default_costs(expr_register: ExprRegister) -> None:
    expr_register.register_cost("l2", lambda dsl: L2Cost())
    expr_register.register_cost("diag_weight", lambda dsl: DiagonalWeightCost(w=np.asarray(dsl["w"], float)))
    expr_register.register_cost("scalar_weight", lambda dsl: ScalarWeightCost(w=float(dsl["w"])))
    expr_register.register_cost("huber", lambda dsl: HuberCost(delta=float(dsl["delta"])))


def create_default_expr_register() -> ExprRegister:
    expr_register = ExprRegister()
    register_stdlib(expr_register)
    register_default_costs(expr_register)
    return expr_register


def build_variable(dsl: dict[str, Any]) -> Variable:
    name = str(dsl["name"])

    if "init" in dsl:
        x = np.asarray(dsl["init"], dtype=float).reshape(-1)
        dim = int(dsl.get("dim", x.size))
        if x.size != dim:
            raise ValueError(f"variable '{name}': init size {x.size} != dim {dim}")
        return Variable(name=name, x=x.copy())

    dim = int(dsl["dim"])
    return Variable(name=name, x=np.zeros((dim,), dtype=float))


def build_variable_pack(dsl: dict[str, Any]) -> VariablePack:
    return VariablePack([build_variable(dsl) for dsl in dsl.get("variables", [])])


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


def build_problem(dsl: dict[str, Any], *, expr_register: ExprRegister) -> tuple[Problem, TimeGrid]:
    time_dsl = dsl.get("time", None)
    time = TimeGrid.single_time() if time_dsl is None else TimeGrid.from_dsl(time_dsl)

    pack = build_variable_pack(dsl)
    env = DslBuildEnv(pack=pack, time=time, expr_register=expr_register, root_dsl=dsl)
    built_terms = [build_term(env, term_dsl) for term_dsl in dsl.get("terms", [])]
    terms = [(expr, cost) for expr, cost, _attrs in built_terms]
    term_attrs = [attrs for _expr, _cost, attrs in built_terms]

    return Problem(variables=pack, terms=terms, term_attrs=term_attrs), time


def collect_required(problem: Problem) -> list[StateKey]:
    return collect_required_from_problem(problem)


def compile_problem(
    dsl: dict[str, Any],
    *,
    build_state: Callable[..., dict],
    expr_register: ExprRegister | None = None,
) -> ProblemRuntime:
    if expr_register is None:
        expr_register = create_default_expr_register()

    problem, time = build_problem(dsl, expr_register=expr_register)
    cache = StateCache(build_state=build_state)
    required = collect_required(problem)
    return ProblemRuntime.from_problem(problem, state=cache, time=time, required=required)
