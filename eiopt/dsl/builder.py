from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..core.state_cache import StateCache, StateKey
from ..core.time_grid import TimeGrid
from ..expr.register_stdlib import register_stdlib
from ..expr.registry import Registry
from ..model.problem import Problem
from ..model.term import (
    DiagonalWeightCost,
    HuberCost,
    L2Cost,
    RuntimeContext,
    ScalarWeightCost,
    Variable,
    VariablePack,
)
from .environment import DslBuildEnv

Array = np.ndarray


def register_default_costs(registry: Registry) -> None:
    registry.register_cost("l2", lambda spec: L2Cost())
    registry.register_cost("diag_weight", lambda spec: DiagonalWeightCost(w=np.asarray(spec["w"], float)))
    registry.register_cost("scalar_weight", lambda spec: ScalarWeightCost(w=float(spec["w"])))
    registry.register_cost("huber", lambda spec: HuberCost(delta=float(spec["delta"])))


def create_default_registry() -> Registry:
    registry = Registry()
    register_stdlib(registry)
    register_default_costs(registry)
    return registry


def build_variable(spec: dict[str, Any]) -> Variable:
    name = str(spec["name"])

    if "init" in spec:
        x = np.asarray(spec["init"], dtype=float).reshape(-1)
        dim = int(spec.get("dim", x.size))
        if x.size != dim:
            raise ValueError(f"variable '{name}': init size {x.size} != dim {dim}")
        return Variable(name=name, x=x.copy())

    dim = int(spec["dim"])
    return Variable(name=name, x=np.zeros((dim,), dtype=float))


def build_variable_pack(dsl: dict[str, Any]) -> VariablePack:
    return VariablePack([build_variable(spec) for spec in dsl.get("variables", [])])


def build_term(env: DslBuildEnv, spec: dict[str, Any]) -> tuple[Any, Any]:
    expr_spec = spec.get("expr", None)
    if not isinstance(expr_spec, dict):
        raise ValueError("term.expr must be a dict.")

    cost_spec = spec.get("cost", {"type": "l2"})
    if not isinstance(cost_spec, dict):
        raise ValueError("term.cost must be a dict.")

    expr = env.build_expr(expr_spec)
    cost = env.build_cost(cost_spec)
    return expr, cost


def build_problem(dsl: dict[str, Any], *, registry: Registry) -> tuple[Problem, TimeGrid]:
    time_spec = dsl.get("time", None)
    time = TimeGrid.single_time() if time_spec is None else TimeGrid.from_dsl(time_spec)

    pack = build_variable_pack(dsl)
    env = DslBuildEnv(pack=pack, time=time, registry=registry)
    terms = [build_term(env, spec) for spec in dsl.get("terms", [])]

    return Problem(variables=pack, terms=terms), time


def collect_required(problem: Problem) -> list[StateKey]:
    required: list[StateKey] = []
    for expr, _cost in problem.terms:
        deps = getattr(expr, "deps", None)
        if callable(deps):
            required.extend(list(deps()))

    return list(dict.fromkeys(required))


def compile_problem(
    dsl: dict[str, Any],
    *,
    build_state: Callable[..., dict],
    registry: Registry | None = None,
) -> tuple[Problem, RuntimeContext, list[StateKey]]:
    if registry is None:
        registry = create_default_registry()

    problem, time = build_problem(dsl, registry=registry)
    cache = StateCache(build_state=build_state)
    ctx = RuntimeContext(
        pack=problem.variables,
        state=cache,
        time=time,
        revision=int(getattr(time, "revision", 0)),
    )
    required = collect_required(problem)
    return problem, ctx, required
