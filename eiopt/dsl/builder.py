from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, List

import numpy as np

from ..core.state_cache import StateCache, StateKey
from ..core.time_grid import TimeGrid
from .context import BuilderContext
from ..expr.registry import Registry
from ..expr.register_stdlib import register_stdlib
from ..model.problem import Problem
from ..model.term import (
    Variable,
    VariablePack,
    EvalContext,
    L2Cost,
    DiagonalWeightCost,
    ScalarWeightCost,
    HuberCost,
)

Array = np.ndarray


def register_default_costs(reg: Registry) -> None:
    reg.register_cost("l2", lambda p: L2Cost())
    reg.register_cost("diag_weight", lambda p: DiagonalWeightCost(w=np.asarray(p["w"], float)))
    reg.register_cost("scalar_weight", lambda p: ScalarWeightCost(w=float(p["w"])))
    reg.register_cost("huber", lambda p: HuberCost(delta=float(p["delta"])))


def create_default_registry() -> Registry:
    reg = Registry()
    register_stdlib(reg)
    register_default_costs(reg)
    return reg


def build_variable(spec: dict) -> Variable:
    name = str(spec["name"])
    if "init" in spec:
        x = np.asarray(spec["init"], dtype=float).reshape(-1)
        dim = int(spec.get("dim", x.size))
        if x.size != dim:
            raise ValueError(f"variable '{name}': init size {x.size} != dim {dim}")
        return Variable(name=name, x=x.copy())
    dim = int(spec["dim"])
    return Variable(name=name, x=np.zeros((dim,), dtype=float))


def build_cost(registry: Registry, spec: dict):
    typ = spec.get("type", "l2")
    fn = registry.cost.get(typ, None)
    if fn is None:
        raise ValueError(f"unknown cost type: {typ}")
    return fn(spec)


def build_expr(ctx: BuilderContext, spec: dict):
    typ = spec["type"]
    fn = ctx.registry.expr.get(typ, None)
    if fn is None:
        raise ValueError(f"unknown expr type: {typ}")
    return fn(ctx, spec)


def build_problem(
    dsl: dict,
    *,
    state_cache: StateCache,
    time: TimeGrid,
    registry: Registry,
    model: Any = None,
) -> tuple[Problem, EvalContext]:
    variables: List[Variable] = [build_variable(v) for v in dsl.get("variables", [])]
    pack = VariablePack(variables)

    ctx_build = BuilderContext(pack=pack, state_cache=state_cache, time=time, registry=registry, model=model)

    terms = []
    for t in dsl.get("terms", []):
        expr = build_expr(ctx_build, t["expr"])
        cost = build_cost(registry, t.get("cost", {"type": "l2"}))
        terms.append((expr, cost))

    problem = Problem(variables=pack, terms=terms)
    ctx_eval = EvalContext(pack=pack, state=state_cache, time=time, revision=int(getattr(time, "revision", 0)))
    return problem, ctx_eval


def collect_required(problem: Problem) -> list[StateKey]:
    req: List[StateKey] = []
    for expr, _cost in problem.terms:
        if hasattr(expr, "deps"):
            req.extend(list(expr.deps()))

    out: List[StateKey] = []
    seen: set[StateKey] = set()
    for k in req:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def prepare_for_solve(
    problem: Problem,
    ctx: EvalContext,
    *,
    required: Optional[Iterable[StateKey]] = None,
) -> list[StateKey]:
    required_list = collect_required(problem) if required is None else list(required)
    if ctx.state is not None and hasattr(ctx.state, "update_if_needed"):
        ctx.state.update_if_needed(ctx.pack, time=ctx.time, required=required_list)
    return required_list


def compile_problem(
    dsl: dict,
    *,
    build_state: Callable[..., dict],
    registry: Registry | None = None,
    model: Any = None,
) -> tuple[Problem, EvalContext, list[StateKey]]:
    if registry is None:
        registry = create_default_registry()

    time_dsl = dsl.get("time", None)
    time = TimeGrid.single_time() if time_dsl is None else TimeGrid.from_dsl(time_dsl)
    cache = StateCache(build_state=build_state)

    problem, ctx = build_problem(dsl, state_cache=cache, time=time, registry=registry, model=model)
    required = collect_required(problem)
    return problem, ctx, required

