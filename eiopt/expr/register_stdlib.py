from __future__ import annotations

import numpy as np

from .registry import Registry
from .nodes import ConstantExpr, GetStateExpr, SubExpr, StackExpr, HingeExpr
from ..core.state_cache import OwnerKey, StateKey
from ..core.state_schema import DEFAULT_FRAME, DTYPE_FRAME, jac_field


def register_stdlib(reg: Registry) -> None:
    reg.register_expr("const", build_const)
    reg.register_expr("get_state", build_get_state)
    reg.register_expr("sub", build_sub)
    reg.register_expr("stack", build_stack)
    reg.register_expr("hinge", build_hinge)


def build_const(ctx, spec):
    if "var" in spec:
        q = next(v for v in ctx.pack.vars if v.name == spec.get("var", "q"))
        return ConstantExpr(
            name=spec.get("name", "const"),
            vars=[q],
            value=np.asarray(spec["value"], float),
        )
    return ConstantExpr(name=spec.get("name", "const"), value=np.asarray(spec["value"], float))


def build_get_state(ctx, spec):
    q = next(v for v in ctx.pack.vars if v.name == spec.get("jac", {}).get("var", "q"))

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

    jac_spec = spec.get("jac", {}) or {}
    jac_var = jac_spec.get("var", "q")
    jac_field_name = jac_spec.get("field", jac_field(field, var=jac_var))
    key_jac = StateKey(k=k, owner=owner, dtype=dtype, field=jac_field_name, frame=frame, rel_frame=rel_frame)

    return GetStateExpr(
        name=spec.get("name", "get_state"),
        vars=[q],
        key_value=key_value,
        key_jac_q=key_jac,
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
