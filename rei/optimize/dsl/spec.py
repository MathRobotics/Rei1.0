from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from ...core.mapping import mapping_as_dict


def load_problem_spec_json(path: str | Path) -> dict[str, Any]:
    """Load a human-oriented problem spec JSON file and convert it to DSL."""

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise TypeError("Problem spec JSON must decode to an object.")
    return problem_spec_to_dsl(data)


def problem_spec_to_dsl(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a compact human-oriented optimization problem spec to Rei DSL.

    The spec intentionally keeps the existing DSL as the compilation target.
    It provides shorthands for common residuals such as ``var - target``,
    ``state - target``, trajectory samples, constraints, and scalar weights.
    Raw DSL nodes can still be embedded with ``{"dsl": {...}}``.
    """

    spec_dict = mapping_as_dict(spec, where="spec")
    dsl: dict[str, Any] = {}

    for key in ("time", "trajectory", "vision"):
        value = spec_dict.get(key, None)
        if value is not None:
            if not isinstance(value, Mapping):
                raise ValueError(f"spec.{key} must be an object.")
            dsl[key] = deepcopy(dict(value))

    variables = _convert_variables(spec_dict.get("variables", []))
    if variables:
        dsl["variables"] = variables

    terms_raw = spec_dict.get("terms", [])
    if not isinstance(terms_raw, Sequence) or isinstance(terms_raw, (str, bytes, bytearray)):
        raise ValueError("spec.terms must be a list.")
    dsl["terms"] = [_convert_term(t, i) for i, t in enumerate(terms_raw)]
    return dsl


def _convert_variables(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        out = []
        for name, value in raw.items():
            entry: dict[str, Any] = {"name": str(name)}
            if isinstance(value, Mapping):
                entry.update(deepcopy(dict(value)))
            elif isinstance(value, int):
                entry["dim"] = int(value)
            else:
                entry["init"] = deepcopy(value)
            out.append(entry)
        return out
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        out = []
        for i, value in enumerate(raw):
            if not isinstance(value, Mapping):
                raise ValueError(f"spec.variables[{i}] must be an object.")
            out.append(deepcopy(dict(value)))
        return out
    raise ValueError("spec.variables must be an object or list.")


def _convert_term(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"spec.terms[{index}] must be an object.")
    term = mapping_as_dict(raw, where=f"spec.terms[{index}]")

    if "dsl" in term:
        dsl_term = term["dsl"]
        if not isinstance(dsl_term, Mapping):
            raise ValueError(f"spec.terms[{index}].dsl must be an object.")
        return deepcopy(dict(dsl_term))

    residual_raw = term.get("residual", term.get("expr", None))
    if residual_raw is None:
        raise ValueError(f"spec.terms[{index}] must contain residual, expr, or dsl.")

    name = str(term.get("name", f"term_{index}"))
    out: dict[str, Any] = {"expr": _convert_residual(residual_raw, name=name, where=f"spec.terms[{index}].residual")}

    cost_raw = term.get("cost", None)
    weight_raw = term.get("weight", term.get("w", None))
    if cost_raw is not None:
        if not isinstance(cost_raw, Mapping):
            raise ValueError(f"spec.terms[{index}].cost must be an object.")
        out["cost"] = deepcopy(dict(cost_raw))
    elif weight_raw is not None:
        out["cost"] = {"type": "scalar_weight", "w": float(weight_raw)}
    else:
        out["cost"] = {"type": "l2"}

    kind = term.get("kind", term.get("constraint", None))
    if kind is not None:
        out["constraint"] = {"kind": _canonical_constraint_kind(kind)}

    attrs = term.get("attrs", None)
    if attrs is not None:
        if not isinstance(attrs, Mapping):
            raise ValueError(f"spec.terms[{index}].attrs must be an object.")
        out["attrs"] = deepcopy(dict(attrs))
    return out


def _convert_residual(raw: Any, *, name: str, where: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be an object.")
    node = mapping_as_dict(raw, where=where)

    if "dsl" in node:
        dsl_node = node["dsl"]
        if not isinstance(dsl_node, Mapping):
            raise ValueError(f"{where}.dsl must be an object.")
        out = deepcopy(dict(dsl_node))
        out.setdefault("name", name)
        return out

    if "op" in node or "type" in node:
        return _convert_op_node(node, name=name, where=where)

    target = node.get("target", node.get("equals", None))
    if target is not None:
        lhs = dict(node)
        lhs.pop("target", None)
        lhs.pop("equals", None)
        return {
            "type": "sub",
            "name": name,
            "a": _convert_leaf(lhs, name=_child_name(name, "value"), where=where),
            "b": _target_to_expr(target, name=_child_name(name, "target"), var=_node_var(node)),
        }

    return _convert_leaf(node, name=name, where=where)


def _convert_op_node(node: Mapping[str, Any], *, name: str, where: str) -> dict[str, Any]:
    op = str(node.get("op", node.get("type"))).strip()
    if op in ("sub", "add"):
        a = node.get("a", None)
        b = node.get("b", None)
        if a is None or b is None:
            raise ValueError(f"{where}: op {op!r} requires a and b.")
        return {
            "type": op,
            "name": str(node.get("name", name)),
            "a": _convert_residual(a, name=_child_name(name, "a"), where=f"{where}.a"),
            "b": _convert_residual(b, name=_child_name(name, "b"), where=f"{where}.b"),
        }
    if op == "hinge":
        base = node.get("base", None)
        if base is None:
            raise ValueError(f"{where}: op 'hinge' requires base.")
        return {
            "type": "hinge",
            "name": str(node.get("name", name)),
            "base": _convert_residual(base, name=_child_name(name, "base"), where=f"{where}.base"),
        }
    if op == "stack":
        range_raw = node.get("range", None)
        inner_raw = node.get("inner", None)
        if not isinstance(range_raw, Mapping):
            raise ValueError(f"{where}: op 'stack' requires range object.")
        if inner_raw is None:
            raise ValueError(f"{where}: op 'stack' requires inner.")
        return {
            "type": "stack",
            "name": str(node.get("name", name)),
            "range": deepcopy(dict(range_raw)),
            "inner": _convert_residual(inner_raw, name=_child_name(name, "k"), where=f"{where}.inner"),
        }
    return _convert_leaf(node, name=name, where=where)


def _convert_leaf(node: Mapping[str, Any], *, name: str, where: str) -> dict[str, Any]:
    if "var" in node and set(node.keys()).issubset({"type", "op", "name", "var"}):
        return {"type": "get_var", "name": str(node.get("name", name)), "var": str(node["var"])}

    if "state" in node:
        state = node["state"]
        state_dsl = dict(state) if isinstance(state, Mapping) else {"name": str(state)}
        out = {
            "type": "get_state",
            "name": str(node.get("name", state_dsl.pop("name", name))),
            "key": _state_key_from_spec(state_dsl, fallback=node),
        }
        var = node.get("var", state_dsl.get("var", None))
        if var is not None:
            out["jac"] = {"var": str(var)}
        return out

    if "traj" in node or "trajectory" in node:
        traj = node.get("traj", node.get("trajectory"))
        traj_dsl = dict(traj) if isinstance(traj, Mapping) else {}
        var = traj_dsl.get("var", node.get("var", None))
        if var is None:
            raise ValueError(f"{where}: trajectory residual requires var.")
        out = {
            "type": "get_traj_var",
            "name": str(node.get("name", traj_dsl.get("name", name))),
            "var": str(var),
        }
        for src, dst in (("at", "k"), ("k", "k"), ("derivative", "derivative_order"), ("derivative_order", "derivative_order")):
            value = node.get(src, traj_dsl.get(src, None))
            if value is not None:
                out[dst] = value
        derivative_wrt = node.get("derivative_wrt", traj_dsl.get("derivative_wrt", None))
        if derivative_wrt is not None:
            out["derivative_wrt"] = str(derivative_wrt)
        return out

    if "const_repeat" in node:
        return _const_expr(node["const_repeat"], name=name, var=node.get("var", None), repeat=True)

    if "const" in node or "value" in node:
        value = node.get("const", node.get("value"))
        repeat = bool(node.get("repeat", False))
        return _const_expr(value, name=str(node.get("name", name)), var=node.get("var", None), repeat=repeat)

    typ = node.get("type", node.get("op", None))
    if typ is not None:
        return deepcopy(dict(node))
    raise ValueError(f"{where}: unsupported residual leaf. Use var, state, traj, const, or dsl.")


def _target_to_expr(target: Any, *, name: str, var: Any) -> dict[str, Any]:
    repeat = False
    value = target
    if isinstance(target, Mapping):
        repeat = bool(target.get("repeat", target.get("repeated", False)))
        if "repeat" in target or "repeated" in target:
            value = {
                k: deepcopy(v)
                for k, v in target.items()
                if k not in ("repeat", "repeated")
            }
    return _const_expr(value, name=name, var=var, repeat=repeat)


def _const_expr(value: Any, *, name: str, var: Any, repeat: bool = False) -> dict[str, Any]:
    out = {
        "type": "const_repeat" if repeat else "const",
        "name": name,
        "value": deepcopy(value),
    }
    if var is not None:
        out["var"] = str(var)
    return out


def _node_var(node: Mapping[str, Any]) -> Any:
    if "var" in node:
        return node.get("var", None)
    traj = node.get("traj", node.get("trajectory", None))
    if isinstance(traj, Mapping) and "var" in traj:
        return traj.get("var", None)
    return None


def _state_key_from_spec(state: Mapping[str, Any], *, fallback: Mapping[str, Any]) -> dict[str, Any]:
    if "key" in state:
        key = state["key"]
        if not isinstance(key, Mapping):
            raise ValueError("state.key must be an object.")
        return deepcopy(dict(key))

    out: dict[str, Any] = {}
    aliases = {
        "at": "k",
        "k": "k",
        "owner_type": "owner_type",
        "owner": "owner_name",
        "owner_name": "owner_name",
        "dtype": "dtype",
        "field": "field",
        "frame": "frame",
        "rel_frame": "rel_frame",
    }
    for src, dst in aliases.items():
        value = state.get(src, fallback.get(src, None))
        if value is not None:
            out[dst] = value
    if "k" not in out:
        out["k"] = fallback.get("at", fallback.get("k", 0))
    missing = [key for key in ("k", "owner_type", "owner_name", "dtype", "field") if key not in out or out[key] is None]
    if missing:
        raise ValueError(f"state residual is missing key field(s): {', '.join(missing)}.")
    return out


def _canonical_constraint_kind(kind: Any) -> str:
    value = str(kind).strip().lower()
    if value in ("eq", "equality"):
        return "eq"
    if value in ("ineq", "inequality"):
        return "ineq"
    raise ValueError(f"constraint kind must be 'eq' or 'ineq'. Got {kind!r}.")


def _child_name(name: str, suffix: str) -> str:
    return f"{name}_{suffix}"


__all__ = [
    "load_problem_spec_json",
    "problem_spec_to_dsl",
]
