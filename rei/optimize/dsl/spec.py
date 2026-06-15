from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.mapping import mapping_as_dict
from .spec_reserved import resolve_opt_vals, resolve_quantity


@dataclass(frozen=True)
class _SpecContext:
    has_trajectory: bool
    trajectory_var: str | None
    joint_var: str
    variables: dict[str, dict[str, Any]]
    time_steps: int | None


def load_problem_spec_toml(path: str | Path) -> dict[str, Any]:
    """Load a human-oriented problem spec TOML file and convert it to DSL."""

    p = Path(path)
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise TypeError("Problem spec TOML must decode to an object.")
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

    opt_vals = resolve_opt_vals(spec_dict.get("opt_vals", None))
    variables = _merge_optimization_variables(
        _convert_variables(_raw_optimization_variables(spec_dict)),
        list(opt_vals.variables),
    )
    if variables:
        dsl["variables"] = variables

    ctx = _build_spec_context(spec_dict=spec_dict, variables=variables)

    terms_raw = spec_dict.get("terms", [])
    if not isinstance(terms_raw, Sequence) or isinstance(terms_raw, (str, bytes, bytearray)):
        raise ValueError("spec.terms must be a list.")
    dsl["terms"] = [
        _convert_term(t, i, var_aliases=opt_vals.aliases, ctx=ctx)
        for i, t in enumerate(terms_raw)
    ]
    return dsl


def _build_spec_context(*, spec_dict: Mapping[str, Any], variables: Sequence[Mapping[str, Any]]) -> _SpecContext:
    trajectory_raw = spec_dict.get("trajectory", None)
    trajectory_var = None
    if isinstance(trajectory_raw, Mapping):
        var_raw = trajectory_raw.get("var", None)
        if var_raw is not None:
            trajectory_var = str(var_raw)

    joint_var = "q"
    joint_raw = spec_dict.get("joint", spec_dict.get("joints", None))
    if isinstance(joint_raw, Mapping):
        var_raw = joint_raw.get("var", joint_raw.get("q_var", None))
        if var_raw is not None:
            joint_var = str(var_raw)
    elif isinstance(joint_raw, str):
        joint_var = str(joint_raw)

    variables_by_name: dict[str, dict[str, Any]] = {}
    for entry in variables:
        if not isinstance(entry, Mapping):
            continue
        name_raw = entry.get("name", None)
        if name_raw is None:
            continue
        variables_by_name[str(name_raw)] = deepcopy(dict(entry))

    time_steps = None
    time_raw = spec_dict.get("time", None)
    if isinstance(time_raw, Mapping) and "N" in time_raw:
        try:
            n_intervals = int(time_raw["N"])
            if n_intervals >= 0:
                time_steps = int(n_intervals + 1)
        except Exception:
            time_steps = None

    return _SpecContext(
        has_trajectory=isinstance(trajectory_raw, Mapping),
        trajectory_var=trajectory_var,
        joint_var=joint_var,
        variables=variables_by_name,
        time_steps=time_steps,
    )


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
                raise ValueError(f"spec optimization variable entry at index {i} must be an object.")
            out.append(deepcopy(dict(value)))
        return out
    raise ValueError("spec.optimization_variables must be an object or list.")


def _raw_optimization_variables(spec: Mapping[str, Any]) -> Any:
    has_preferred = "optimization_variables" in spec
    has_legacy = "variables" in spec
    if has_preferred and has_legacy:
        raise ValueError("spec must not contain both optimization_variables and variables.")
    if has_preferred:
        return spec.get("optimization_variables", [])
    return spec.get("variables", [])


def _merge_optimization_variables(
    explicit_variables: list[dict[str, Any]],
    opt_val_variables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in [*explicit_variables, *opt_val_variables]:
        name = str(entry.get("name", ""))
        if name == "":
            raise ValueError("optimization variable entries must define a non-empty name.")
        if name in seen:
            raise ValueError(f"optimization variable {name!r} is defined more than once.")
        seen.add(name)
        out.append(entry)
    return out


def _convert_term(
    raw: Any,
    index: int,
    *,
    var_aliases: Mapping[str, str],
    ctx: _SpecContext,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"spec.terms[{index}] must be an object.")
    term = mapping_as_dict(raw, where=f"spec.terms[{index}]")

    if "dsl" in term:
        dsl_term = term["dsl"]
        if not isinstance(dsl_term, Mapping):
            raise ValueError(f"spec.terms[{index}].dsl must be an object.")
        return deepcopy(dict(dsl_term))

    residual_raw = _term_residual_raw(term)
    if residual_raw is None:
        residual_raw = _term_shorthand_residual(term)
    if residual_raw is None:
        raise ValueError(f"spec.terms[{index}] must contain residual, expr, dsl, or a term shorthand.")

    name = str(term.get("name", f"term_{index}"))
    out: dict[str, Any] = {
        "expr": _convert_residual(
            residual_raw,
            name=name,
            where=f"spec.terms[{index}].residual",
            var_aliases=var_aliases,
            ctx=ctx,
        )
    }

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
    attrs_out: dict[str, Any] = {}
    if attrs is not None:
        if not isinstance(attrs, Mapping):
            raise ValueError(f"spec.terms[{index}].attrs must be an object.")
        attrs_out.update(deepcopy(dict(attrs)))
    for key in _TERM_SHORTHAND_ATTR_KEYS:
        if key in term:
            value = term[key]
            if key == "plot":
                value = _convert_plot_metadata(value, term=term)
            attrs_out[key] = deepcopy(value)
    if attrs_out:
        out["attrs"] = attrs_out
    return out


def _convert_plot_metadata(raw: Any, *, term: Mapping[str, Any]) -> Any:
    if "quantity" not in term:
        return deepcopy(raw)

    derivative_order = _quantity_derivative_order(term["quantity"])
    if derivative_order is None:
        return deepcopy(raw)

    def convert_one(item: Any) -> Any:
        if isinstance(item, bool):
            if not item:
                return False
            return {
                "type": "traj_derivative",
                "derivative_order": int(derivative_order),
            }
        if isinstance(item, str):
            name = item.strip()
            if name == "":
                return item
            return {
                "type": "traj_derivative",
                "name": name,
                "derivative_order": int(derivative_order),
            }
        if isinstance(item, Mapping):
            out = deepcopy(dict(item))
            out.setdefault("type", "traj_derivative")
            if "derivative_order" not in out and "order" not in out:
                out["derivative_order"] = int(derivative_order)
            return out
        return deepcopy(item)

    if isinstance(raw, list):
        return [convert_one(item) for item in raw]
    return convert_one(raw)


def _quantity_derivative_order(raw: Any) -> int | None:
    try:
        expanded = resolve_quantity(raw)
    except Exception:
        return None
    traj = expanded.get("traj", None)
    if not isinstance(traj, Mapping):
        return None
    order = traj.get("derivative", traj.get("derivative_order", None))
    if order is None:
        return None
    return int(order)


def _term_residual_raw(term: Mapping[str, Any]) -> Any:
    residual_raw = term.get("residual", term.get("expr", None))
    if residual_raw is not None:
        return residual_raw

    term_type = str(term.get("type", term.get("kind_type", ""))).strip().lower()
    if term_type in ("joint", "joint_angle", "joint_angles", "joint_target", "joint_q"):
        return _copy_selected_term_fields(
            term,
            selected=(
                "joint",
                "var",
                "at",
                "k",
                "derivative",
                "derivative_order",
                "derivative_wrt",
                "target",
                "equals",
                "dim",
                "segment_dim",
            ),
            defaults={"joint": True},
        )

    joint_raw = term.get("joint", None)
    if joint_raw is not None:
        if isinstance(joint_raw, Mapping):
            residual = deepcopy(dict(joint_raw))
        elif isinstance(joint_raw, bool):
            residual = {"joint": bool(joint_raw)}
        else:
            residual = {"joint": joint_raw}
        for key in ("var", "at", "k", "target", "equals", "dim", "segment_dim"):
            if key in term and key not in residual:
                residual[key] = deepcopy(term[key])
        return residual

    return None


def _copy_selected_term_fields(
    term: Mapping[str, Any],
    *,
    selected: Sequence[str],
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    out = deepcopy(dict(defaults))
    for key in selected:
        if key in term:
            out[key] = deepcopy(term[key])
    return out


def _convert_residual(
    raw: Any,
    *,
    name: str,
    where: str,
    var_aliases: Mapping[str, str],
    ctx: _SpecContext,
) -> dict[str, Any]:
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
        return _convert_op_node(node, name=name, where=where, var_aliases=var_aliases, ctx=ctx)

    if "bounds" in node:
        return _convert_bounds_node(node, name=name, where=where, var_aliases=var_aliases, ctx=ctx)

    target = node.get("target", node.get("equals", None))
    if target is not None:
        lhs = dict(node)
        lhs.pop("target", None)
        lhs.pop("equals", None)
        return {
            "type": "sub",
            "name": name,
            "a": _convert_leaf(lhs, name=_child_name(name, "value"), where=where, var_aliases=var_aliases, ctx=ctx),
            "b": _target_to_expr(
                target,
                name=_child_name(name, "target"),
                var=_node_var(node, var_aliases=var_aliases, ctx=ctx),
                dim=_target_dim_hint(node, var_aliases=var_aliases, ctx=ctx),
            ),
        }

    return _convert_leaf(node, name=name, where=where, var_aliases=var_aliases, ctx=ctx)


def _convert_bounds_node(
    node: Mapping[str, Any],
    *,
    name: str,
    where: str,
    var_aliases: Mapping[str, str],
    ctx: _SpecContext,
) -> dict[str, Any]:
    bounds_raw = node.get("bounds", None)
    if not isinstance(bounds_raw, Mapping):
        raise ValueError(f"{where}.bounds must be an object.")
    bounds = mapping_as_dict(bounds_raw, where=f"{where}.bounds")

    allowed = {"lower", "upper"}
    unknown = sorted(str(k) for k in bounds if str(k) not in allowed)
    if unknown:
        raise ValueError(
            f"{where}.bounds has unsupported key(s): {', '.join(unknown)}. "
            "Use lower and/or upper."
        )

    subject_node = {
        key: deepcopy(value)
        for key, value in node.items()
        if key not in ("bounds", "target", "equals")
    }
    if not subject_node:
        raise ValueError(f"{where}.bounds requires a bounded quantity, state, trajectory, or variable.")

    var = _node_var(subject_node, var_aliases=var_aliases, ctx=ctx)

    parts: list[dict[str, Any]] = []
    if "upper" in bounds:
        subject = _convert_leaf(
            subject_node,
            name=_child_name(name, "upper_value"),
            where=f"{where}.bounds.upper.value",
            var_aliases=var_aliases,
            ctx=ctx,
        )
        upper = _const_expr(
            bounds["upper"],
            name=_child_name(name, "upper"),
            var=var,
            repeat=True,
        )
        parts.append(
            {
                "type": "hinge",
                "name": _child_name(name, "upper_violation"),
                "base": {
                    "type": "sub",
                    "name": _child_name(name, "upper_margin"),
                    "a": subject,
                    "b": upper,
                },
            }
        )

    if "lower" in bounds:
        lower = _const_expr(
            bounds["lower"],
            name=_child_name(name, "lower"),
            var=var,
            repeat=True,
        )
        subject = _convert_leaf(
            subject_node,
            name=_child_name(name, "lower_value"),
            where=f"{where}.bounds.lower.value",
            var_aliases=var_aliases,
            ctx=ctx,
        )
        parts.append(
            {
                "type": "hinge",
                "name": _child_name(name, "lower_violation"),
                "base": {
                    "type": "sub",
                    "name": _child_name(name, "lower_margin"),
                    "a": lower,
                    "b": subject,
                },
            }
        )

    if not parts:
        raise ValueError(f"{where}.bounds must define lower and/or upper.")
    if len(parts) == 1:
        return parts[0]
    return {
        "type": "vstack",
        "name": name,
        "parts": parts,
    }


def _convert_op_node(
    node: Mapping[str, Any],
    *,
    name: str,
    where: str,
    var_aliases: Mapping[str, str],
    ctx: _SpecContext,
) -> dict[str, Any]:
    op = str(node.get("op", node.get("type"))).strip()
    if op in ("sub", "add"):
        a = node.get("a", None)
        b = node.get("b", None)
        if a is None or b is None:
            raise ValueError(f"{where}: op {op!r} requires a and b.")
        return {
            "type": op,
            "name": str(node.get("name", name)),
            "a": _convert_residual(
                a,
                name=_child_name(name, "a"),
                where=f"{where}.a",
                var_aliases=var_aliases,
                ctx=ctx,
            ),
            "b": _convert_residual(
                b,
                name=_child_name(name, "b"),
                where=f"{where}.b",
                var_aliases=var_aliases,
                ctx=ctx,
            ),
        }
    if op == "hinge":
        base = node.get("base", None)
        if base is None:
            raise ValueError(f"{where}: op 'hinge' requires base.")
        return {
            "type": "hinge",
            "name": str(node.get("name", name)),
            "base": _convert_residual(
                base,
                name=_child_name(name, "base"),
                where=f"{where}.base",
                var_aliases=var_aliases,
                ctx=ctx,
            ),
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
            "inner": _convert_residual(
                inner_raw,
                name=_child_name(name, "k"),
                where=f"{where}.inner",
                var_aliases=var_aliases,
                ctx=ctx,
            ),
        }
    return _convert_leaf(node, name=name, where=where, var_aliases=var_aliases, ctx=ctx)


def _convert_leaf(
    node: Mapping[str, Any],
    *,
    name: str,
    where: str,
    var_aliases: Mapping[str, str],
    ctx: _SpecContext,
) -> dict[str, Any]:
    if "joint" in node:
        return _convert_joint_leaf(node, name=name, where=where, var_aliases=var_aliases, ctx=ctx)

    if "quantity" in node:
        quantity_overrides = {
            key: deepcopy(value)
            for key, value in node.items()
            if key not in ("quantity", "target", "equals")
        }
        quantity_overrides.setdefault("name", name)
        expanded = resolve_quantity(node["quantity"], overrides=quantity_overrides)
        return _convert_residual(expanded, name=name, where=where, var_aliases=var_aliases, ctx=ctx)

    if "var" in node and set(node.keys()).issubset({"type", "op", "name", "var"}):
        return {
            "type": "get_var",
            "name": str(node.get("name", name)),
            "var": _resolve_var_alias(node["var"], var_aliases=var_aliases),
        }

    if "state" in node:
        state = node["state"]
        state_dsl = _state_spec_from_raw(state, where=f"{where}.state")
        out = {
            "type": "get_state",
            "name": str(node.get("name", state_dsl.pop("name", name))),
            "key": _state_key_from_spec(state_dsl, fallback=node),
        }
        var = node.get("var", state_dsl.get("var", None))
        if var is not None:
            out["jac"] = {"var": _resolve_var_alias(var, var_aliases=var_aliases)}
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
            "var": _resolve_var_alias(var, var_aliases=var_aliases),
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
        return _const_expr(
            node["const_repeat"],
            name=name,
            var=_resolve_optional_var_alias(node.get("var", None), var_aliases=var_aliases),
            repeat=True,
        )

    if "const" in node or "value" in node:
        value = node.get("const", node.get("value"))
        repeat = bool(node.get("repeat", False))
        return _const_expr(
            value,
            name=str(node.get("name", name)),
            var=_resolve_optional_var_alias(node.get("var", None), var_aliases=var_aliases),
            repeat=repeat,
        )

    typ = node.get("type", node.get("op", None))
    if typ is not None:
        return deepcopy(dict(node))
    raise ValueError(f"{where}: unsupported residual leaf. Use var, state, traj, const, or dsl.")


def _convert_joint_leaf(
    node: Mapping[str, Any],
    *,
    name: str,
    where: str,
    var_aliases: Mapping[str, str],
    ctx: _SpecContext,
) -> dict[str, Any]:
    joint_raw = node.get("joint", True)
    joint_dsl = dict(joint_raw) if isinstance(joint_raw, Mapping) else {}

    if ctx.has_trajectory:
        var = node.get("var", joint_dsl.get("var", ctx.trajectory_var))
        if var is None:
            raise ValueError(f"{where}: joint residual with trajectory requires trajectory.var or joint.var.")
        out = {
            "type": "get_traj_var",
            "name": str(node.get("name", joint_dsl.get("name", name))),
            "var": _resolve_var_alias(var, var_aliases=var_aliases),
        }
        for src, dst in (("at", "k"), ("k", "k"), ("derivative", "derivative_order"), ("derivative_order", "derivative_order")):
            value = node.get(src, joint_dsl.get(src, None))
            if value is not None:
                out[dst] = value
        derivative_wrt = node.get("derivative_wrt", joint_dsl.get("derivative_wrt", None))
        if derivative_wrt is not None:
            out["derivative_wrt"] = str(derivative_wrt)
        return out

    joint_var_from_token = joint_raw if isinstance(joint_raw, str) else None
    var = node.get("var", joint_dsl.get("var", joint_var_from_token or ctx.joint_var))
    if var is None:
        raise ValueError(f"{where}: joint residual requires joint.var or residual.var.")
    out = {
        "type": "get_var",
        "name": str(node.get("name", joint_dsl.get("name", name))),
        "var": _resolve_var_alias(var, var_aliases=var_aliases),
    }
    k = node.get("at", joint_dsl.get("at", node.get("k", joint_dsl.get("k", None))))
    if k is not None:
        out["k"] = k
    return out


_TERM_SHORTHAND_METADATA_KEYS = {
    "name",
    "residual",
    "expr",
    "dsl",
    "type",
    "kind_type",
    "cost",
    "weight",
    "w",
    "kind",
    "constraint",
    "attrs",
    "enforce",
    "plot",
}
_TERM_SHORTHAND_ATTR_KEYS = {
    "enforce",
    "plot",
}
_TERM_SHORTHAND_LEAF_KEYS = {
    "var",
    "state",
    "traj",
    "trajectory",
    "const",
    "value",
    "const_repeat",
    "quantity",
    "joint",
    "bounds",
    "target",
    "equals",
    "dim",
    "segment_dim",
    "repeat",
    "at",
    "k",
    "derivative",
    "derivative_order",
    "derivative_wrt",
    "k0",
    "k1",
    "stride",
    "owner_type",
    "owner",
    "owner_name",
    "dtype",
    "field",
    "frame",
    "rel_frame",
}


def _term_shorthand_residual(term: Mapping[str, Any]) -> dict[str, Any] | None:
    if not any(key in term for key in _TERM_SHORTHAND_LEAF_KEYS):
        return None
    residual: dict[str, Any] = {}
    for key, value in term.items():
        key_str = str(key)
        if key_str in _TERM_SHORTHAND_METADATA_KEYS:
            continue
        if key_str in _TERM_SHORTHAND_LEAF_KEYS:
            residual[key_str] = deepcopy(value)
            continue
        raise ValueError(
            "term shorthand contains unsupported key "
            f"{key_str!r}. Move advanced expressions under `residual` or `dsl`."
        )
    return residual


def _state_spec_from_raw(raw: Any, *, where: str) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return deepcopy(dict(raw))

    text = str(raw)
    parsed = _parse_dotted_state_ref(text, where=where)
    if parsed is not None:
        return parsed
    return {"name": text}


def _parse_dotted_state_ref(text: str, *, where: str) -> dict[str, Any] | None:
    if "." not in text:
        return None
    parts = text.split(".")
    if len(parts) != 4 or any(part == "" for part in parts):
        raise ValueError(
            f"{where}: dotted state shorthand must be "
            "`<dtype>.<owner_type>.<owner_name>.<field>`, got {text!r}."
        )
    dtype, owner_type, owner_name, field = parts
    return {
        "dtype": dtype,
        "owner_type": owner_type,
        "owner_name": owner_name,
        "field": field,
    }


def _target_to_expr(target: Any, *, name: str, var: Any, dim: int | None = None) -> dict[str, Any]:
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
    return _const_expr(value, name=name, var=var, repeat=repeat, dim=dim)


def _const_expr(value: Any, *, name: str, var: Any, repeat: bool = False, dim: int | None = None) -> dict[str, Any]:
    out = {
        "type": "const_repeat" if repeat else "const",
        "name": name,
        "value": deepcopy(value),
    }
    if var is not None:
        out["var"] = str(var)
    if dim is not None:
        out["dim"] = int(dim)
    return out


def _resolve_var_alias(value: Any, *, var_aliases: Mapping[str, str]) -> str:
    text = str(value)
    return str(var_aliases.get(text, text))


def _resolve_optional_var_alias(value: Any, *, var_aliases: Mapping[str, str]) -> str | None:
    if value is None:
        return None
    return _resolve_var_alias(value, var_aliases=var_aliases)


def _node_var(node: Mapping[str, Any], *, var_aliases: Mapping[str, str], ctx: _SpecContext) -> Any:
    if "var" in node:
        return _resolve_optional_var_alias(node.get("var", None), var_aliases=var_aliases)
    traj = node.get("traj", node.get("trajectory", None))
    if isinstance(traj, Mapping) and "var" in traj:
        return _resolve_optional_var_alias(traj.get("var", None), var_aliases=var_aliases)
    if "quantity" in node:
        quantity_overrides = {
            key: deepcopy(value)
            for key, value in node.items()
            if key not in ("quantity", "target", "equals")
        }
        expanded = resolve_quantity(node["quantity"], overrides=quantity_overrides)
        return _node_var(expanded, var_aliases=var_aliases, ctx=ctx)
    if "joint" in node:
        joint = node.get("joint", None)
        if isinstance(joint, Mapping) and "var" in joint:
            return _resolve_optional_var_alias(joint.get("var", None), var_aliases=var_aliases)
        if isinstance(joint, str) and not ctx.has_trajectory:
            return _resolve_var_alias(joint, var_aliases=var_aliases)
        var = ctx.trajectory_var if ctx.has_trajectory else ctx.joint_var
        return _resolve_optional_var_alias(var, var_aliases=var_aliases)
    return None


def _target_dim_hint(node: Mapping[str, Any], *, var_aliases: Mapping[str, str], ctx: _SpecContext) -> int | None:
    dim_raw = node.get("dim", node.get("segment_dim", None))
    if dim_raw is not None:
        try:
            dim = int(dim_raw)
        except Exception as e:
            raise ValueError(f"target dim must be an integer, got {dim_raw!r}.") from e
        if dim <= 0:
            raise ValueError(f"target dim must be > 0, got {dim}.")
        return dim

    if "joint" not in node or ctx.has_trajectory:
        return None

    var_name = _node_var(node, var_aliases=var_aliases, ctx=ctx)
    if var_name is None:
        return None
    var_dim = _variable_dim(ctx.variables.get(str(var_name), None))
    if var_dim is None:
        return None

    k = node.get("at", node.get("k", None))
    joint = node.get("joint", None)
    if isinstance(joint, Mapping):
        k = node.get("at", joint.get("at", node.get("k", joint.get("k", k))))
    if k is None or ctx.time_steps is None or ctx.time_steps <= 1:
        return int(var_dim)
    if int(var_dim) % int(ctx.time_steps) != 0:
        return int(var_dim)
    return int(var_dim) // int(ctx.time_steps)


def _variable_dim(var_dsl: Mapping[str, Any] | None) -> int | None:
    if var_dsl is None:
        return None
    dim_raw = var_dsl.get("dim", None)
    if dim_raw is not None:
        try:
            dim = int(dim_raw)
            return dim if dim > 0 else None
        except Exception:
            return None
    init = var_dsl.get("init", None)
    if isinstance(init, Mapping):
        return None
    if init is None:
        return None
    if isinstance(init, Sequence) and not isinstance(init, (str, bytes, bytearray)):
        return len(list(init))
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
    "load_problem_spec_toml",
    "problem_spec_to_dsl",
]
