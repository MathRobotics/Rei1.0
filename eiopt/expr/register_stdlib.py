from __future__ import annotations

import numpy as np

from .expr_register import ExprRegister
from .nodes import ConstantExpr, GetStateExpr, GetVarExpr, SubExpr, StackExpr, HingeExpr, TrajectoryVarExpr, TimeDiffExpr
from ..core.state_cache import OwnerKey, StateKey
from ..core.state_schema import DEFAULT_FRAME, DTYPE_KINEMATICS, jac_field
from ..dsl.trajectory import build_trajectory_map, default_steps_from_time, infer_bspline_q_dim_from_var


def register_stdlib(expr_register: ExprRegister) -> None:
    expr_register.register_expr("const", build_const)
    expr_register.register_expr("get_state", build_get_state)
    expr_register.register_expr("get_var", build_get_var)
    expr_register.register_expr("get_traj_var", build_get_traj_var)
    expr_register.register_expr("sub", build_sub)
    expr_register.register_expr("stack", build_stack)
    expr_register.register_expr("hinge", build_hinge)
    expr_register.register_expr("time_diff", build_time_diff)


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


def build_const(ctx, dsl):
    if "var" in dsl:
        var_name = str(dsl.get("var", _default_var_name(ctx)))
        q = next((v for v in ctx.pack.vars if v.name == var_name), None)
        if q is None:
            raise ValueError(f"const: unknown variable: {var_name!r}")
        return ConstantExpr(
            name=dsl.get("name", "const"),
            vars=[q],
            value=np.asarray(dsl["value"], float),
        )
    return ConstantExpr(name=dsl.get("name", "const"), value=np.asarray(dsl["value"], float))


def build_get_state(ctx, dsl):
    jac_dsl = dsl.get("jac", {}) or {}
    jac_var = jac_dsl.get("var", None)
    if jac_var is None:
        jac_var = _default_var_name(ctx)
    jac_var = str(jac_var)

    q = next((v for v in ctx.pack.vars if v.name == jac_var), None)
    if q is None:
        raise ValueError(f"get_state: unknown jac variable: {jac_var!r}")

    key_dsl = dsl["key"]
    k = int(key_dsl.get("k", 0))
    owner = OwnerKey(key_dsl["owner_type"], key_dsl["owner_name"])
    dtype = str(key_dsl["dtype"])
    field = key_dsl["field"]
    frame = key_dsl.get("frame", None)
    rel_frame = key_dsl.get("rel_frame", None)

    if dtype == DTYPE_KINEMATICS:
        if frame is None:
            frame = DEFAULT_FRAME
        elif frame != DEFAULT_FRAME:
            raise ValueError(f"get_state: currently only frame='{DEFAULT_FRAME}' is supported (got {frame!r})")

    key_value = StateKey(k=k, owner=owner, dtype=dtype, field=field, frame=frame, rel_frame=rel_frame)

    jac_field_name = jac_dsl.get("field", jac_field(field, var=jac_var))
    key_jac = StateKey(k=k, owner=owner, dtype=dtype, field=jac_field_name, frame=frame, rel_frame=rel_frame)

    return GetStateExpr(
        name=dsl.get("name", "get_state"),
        vars=[q],
        key_value=key_value,
        key_jac_q=key_jac,
    )


def build_get_var(ctx, dsl):
    var_name = dsl.get("var", None)
    if var_name is None:
        var_name = _default_var_name(ctx)
    var_name = str(var_name)

    v = next((x for x in ctx.pack.vars if x.name == var_name), None)
    if v is None:
        raise ValueError(f"get_var: unknown variable: {var_name!r}")

    k = dsl.get("k", None)
    if k is None:
        key = dsl.get("key", None)
        if isinstance(key, dict) and "k" in key:
            k = key.get("k", None)
    k_i = None if k is None else int(k)

    return GetVarExpr(
        name=dsl.get("name", "get_var"),
        vars=[v],
        k=k_i,
    )


def build_get_traj_var(ctx, dsl):
    traj_dsl = dsl.get("trajectory", None)
    if traj_dsl is None:
        root_dsl = getattr(ctx, "root_dsl", None)
        if isinstance(root_dsl, dict):
            traj_dsl = root_dsl.get("trajectory", None)
    if not isinstance(traj_dsl, dict):
        raise ValueError(
            "get_traj_var: trajectory config not found. "
            "Set expr.trajectory or top-level [trajectory]."
        )

    var_name = dsl.get("var", traj_dsl.get("var", None))
    if var_name is None:
        var_name = _default_var_name(ctx, preferred="p")
    var_name = str(var_name)
    if var_name == "":
        raise ValueError("get_traj_var: var must be non-empty.")

    v = next((x for x in ctx.pack.vars if x.name == var_name), None)
    if v is None:
        raise ValueError(f"get_traj_var: unknown variable: {var_name!r}")

    default_q_dim = None
    if str(traj_dsl.get("type", "")).strip().lower() == "bspline":
        default_q_dim = infer_bspline_q_dim_from_var(traj_dsl, var_dim=v.dim())

    resolve_traj = getattr(ctx, "resolve_trajectory_map", None)
    if callable(resolve_traj):
        traj = resolve_traj(traj_dsl, default_q_dim=default_q_dim)
    else:
        traj = build_trajectory_map(
            traj_dsl,
            default_steps=default_steps_from_time(getattr(ctx, "time", None)),
            default_q_dim=default_q_dim,
        )
    if traj.p_dim != v.dim():
        raise ValueError(
            "get_traj_var: variable dimension mismatch with trajectory parameter dimension. "
            f"var {var_name!r} dim={v.dim()}, trajectory p_dim={traj.p_dim}."
        )

    return TrajectoryVarExpr(
        name=dsl.get("name", "get_traj_var"),
        vars=[v],
        trajectory=traj,
    )


def build_sub(ctx, dsl):
    a = ctx.build_expr(dsl["a"])
    b = ctx.build_expr(dsl["b"])
    return SubExpr(name=dsl.get("name", "sub"), a=a, b=b)


def build_stack(ctx, dsl):
    r = dsl["range"]
    k0, k1 = int(r["k0"]), int(r["k1"])
    inner = dsl["inner"]
    parts = []
    for k in range(k0, k1 + 1):
        inner_k = dict(inner)
        inner_k.setdefault("key", dict(inner.get("key", {})))
        inner_k["key"]["k"] = k
        parts.append(ctx.build_expr(inner_k))
    return StackExpr(name=dsl.get("name", "stack"), parts=parts)


def build_hinge(ctx, dsl):
    base = ctx.build_expr(dsl["base"])
    return HingeExpr(name=dsl.get("name", "hinge"), base=base)


def build_time_diff(ctx, dsl):
    base = ctx.build_expr(dsl["base"])
    segment_dim_raw = dsl.get("segment_dim", None)
    if segment_dim_raw is None:
        raise ValueError("time_diff: segment_dim is required.")
    try:
        segment_dim = int(segment_dim_raw)
    except Exception as e:
        raise ValueError(f"time_diff: segment_dim must be an integer, got {segment_dim_raw!r}.") from e
    if segment_dim <= 0:
        raise ValueError(f"time_diff: segment_dim must be > 0, got {segment_dim}.")
    return TimeDiffExpr(
        name=dsl.get("name", "time_diff"),
        base=base,
        segment_dim=segment_dim,
    )
