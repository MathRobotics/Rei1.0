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


def build_term(env: DslBuildEnv, dsl: dict[str, Any]) -> tuple[Any, Any]:
    expr_dsl = dsl.get("expr", None)
    if not isinstance(expr_dsl, dict):
        raise ValueError("term.expr must be a dict.")

    cost_dsl = dsl.get("cost", {"type": "l2"})
    if not isinstance(cost_dsl, dict):
        raise ValueError("term.cost must be a dict.")

    expr = env.build_expr(expr_dsl)
    cost = env.build_cost(cost_dsl)
    return expr, cost


def build_problem(dsl: dict[str, Any], *, expr_register: ExprRegister) -> tuple[Problem, TimeGrid]:
    time_dsl = dsl.get("time", None)
    time = TimeGrid.single_time() if time_dsl is None else TimeGrid.from_dsl(time_dsl)

    pack = build_variable_pack(dsl)
    env = DslBuildEnv(pack=pack, time=time, expr_register=expr_register, root_dsl=dsl)
    terms = [build_term(env, term_dsl) for term_dsl in dsl.get("terms", [])]

    return Problem(variables=pack, terms=terms), time


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
