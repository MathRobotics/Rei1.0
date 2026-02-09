from __future__ import annotations

import numpy as np

from .registry import Registry
from .nodes import ConstantExpr, GetStateExpr, GetVarExpr, SubExpr, StackExpr, HingeExpr
from ..core.state_cache import OwnerKey, StateKey
from ..core.state_schema import DEFAULT_FRAME, DTYPE_FRAME, jac_field


def register_stdlib(reg: Registry) -> None:
    reg.register_expr("const", build_const)
    reg.register_expr("get_state", build_get_state)
    reg.register_expr("get_var", build_get_var)
    reg.register_expr("sub", build_sub)
    reg.register_expr("stack", build_stack)
    reg.register_expr("hinge", build_hinge)


def _default_var_name(ctx, *, preferred: str = "q") -> str:
    names = [v.name for v in ctx.pack.vars]
    if len(names) == 1:
        return names[0]
    if preferred in names:
        return preferred
    raise ValueError(
        "Multiple variables exist; specify the variable explicitly "
        "(e.g. get_var.var='x' or get_state.jac.var='x')."
    )


def build_const(ctx, spec):
    if "var" in spec:
        var_name = str(spec.get("var", _default_var_name(ctx)))
        q = next((v for v in ctx.pack.vars if v.name == var_name), None)
        if q is None:
            raise ValueError(f"const: unknown variable: {var_name!r}")
        return ConstantExpr(
            name=spec.get("name", "const"),
            vars=[q],
            value=np.asarray(spec["value"], float),
        )
    return ConstantExpr(name=spec.get("name", "const"), value=np.asarray(spec["value"], float))


def build_get_state(ctx, spec):
    jac_spec = spec.get("jac", {}) or {}
    jac_var = jac_spec.get("var", None)
    if jac_var is None:
        jac_var = _default_var_name(ctx)
    jac_var = str(jac_var)

    q = next((v for v in ctx.pack.vars if v.name == jac_var), None)
    if q is None:
        raise ValueError(f"get_state: unknown jac variable: {jac_var!r}")

    key_spec = spec["key"]
    k = int(key_spec.get("k", 0))
    owner = OwnerKey(key_spec["owner_type"], key_spec["owner_name"])
    dtype = key_spec["dtype"]
    field = key_spec["field"]
    frame = key_spec.get("frame", None)
    rel_frame = key_spec.get("rel_frame", None)

    if dtype == DTYPE_FRAME:
        if frame is None:
            frame = DEFAULT_FRAME
        elif frame != DEFAULT_FRAME:
            raise ValueError(f"get_state: currently only frame='{DEFAULT_FRAME}' is supported (got {frame!r})")

    key_value = StateKey(k=k, owner=owner, dtype=dtype, field=field, frame=frame, rel_frame=rel_frame)

    jac_field_name = jac_spec.get("field", jac_field(field, var=jac_var))
    key_jac = StateKey(k=k, owner=owner, dtype=dtype, field=jac_field_name, frame=frame, rel_frame=rel_frame)

    return GetStateExpr(
        name=spec.get("name", "get_state"),
        vars=[q],
        key_value=key_value,
        key_jac_q=key_jac,
    )


def build_get_var(ctx, spec):
    var_name = spec.get("var", None)
    if var_name is None:
        var_name = _default_var_name(ctx)
    var_name = str(var_name)

    v = next((x for x in ctx.pack.vars if x.name == var_name), None)
    if v is None:
        raise ValueError(f"get_var: unknown variable: {var_name!r}")

    k = spec.get("k", None)
    if k is None:
        key = spec.get("key", None)
        if isinstance(key, dict) and "k" in key:
            k = key.get("k", None)
    k_i = None if k is None else int(k)

    return GetVarExpr(
        name=spec.get("name", "get_var"),
        vars=[v],
        k=k_i,
    )


def build_sub(ctx, spec):
    a = ctx.registry.expr[spec["a"]["type"]](ctx, spec["a"])
    b = ctx.registry.expr[spec["b"]["type"]](ctx, spec["b"])
    return SubExpr(name=spec.get("name", "sub"), a=a, b=b)


def build_stack(ctx, spec):
    r = spec["range"]
    k0, k1 = int(r["k0"]), int(r["k1"])
    inner = spec["inner"]
    parts = []
    for k in range(k0, k1 + 1):
        inner_k = dict(inner)
        inner_k.setdefault("key", dict(inner.get("key", {})))
        inner_k["key"]["k"] = k
        parts.append(ctx.registry.expr[inner_k["type"]](ctx, inner_k))
    return StackExpr(name=spec.get("name", "stack"), parts=parts)


def build_hinge(ctx, spec):
    base = ctx.registry.expr[spec["base"]["type"]](ctx, spec["base"])
    return HingeExpr(name=spec.get("name", "hinge"), base=base)
