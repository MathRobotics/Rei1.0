from __future__ import annotations

import numpy as np

from .expr_register import ExprRegister
from .nodes import (
    ConstantExpr,
    RepeatConstantExpr,
    GetStateExpr,
    GetVarExpr,
    SubExpr,
    StackExpr,
    HingeExpr,
    TrajectoryVarExpr,
    TrajectoryVarDerivativesExpr,
    TimeDiffExpr,
)
from ..core.state_cache import OwnerKey, StateKey
from ..core.state_schema import DEFAULT_FRAME, DTYPE_KINEMATICS, canonical_dtype_name, canonical_field_name, jac_field
from ..dsl.trajectory import (
    build_trajectory_map,
    build_trajectory_map_with_derivative,
    build_trajectory_maps_with_derivatives,
    default_dt_from_time,
    default_steps_from_time,
    infer_bspline_q_dim_from_var,
)


def register_stdlib(expr_register: ExprRegister) -> None:
    expr_register.register_expr("const", build_const)
    expr_register.register_expr("const_repeat", build_const_repeat)
    expr_register.register_expr("get_state", build_get_state)
    expr_register.register_expr("get_var", build_get_var)
    expr_register.register_expr("get_traj_var", build_get_traj_var)
    expr_register.register_expr("sub", build_sub)
    expr_register.register_expr("stack", build_stack)
    expr_register.register_expr("hinge", build_hinge)
    expr_register.register_expr("time_diff", build_time_diff)


def _default_var_name(ctx, *, preferred: str = "q") -> str:
    del preferred
    names = [v.name for v in ctx.pack.vars]
    if len(names) == 1:
        return names[0]
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


def build_const_repeat(ctx, dsl):
    repeats_raw = dsl.get("repeats", dsl.get("steps", None))
    if repeats_raw is None:
        time = getattr(ctx, "time", None)
        if time is not None and hasattr(time, "N"):
            try:
                repeats_raw = int(time.N) + 1
            except Exception:
                repeats_raw = None
    if repeats_raw is None:
        raise ValueError("const_repeat: repeats (or steps) is required when time.N is unavailable.")

    try:
        repeats = int(repeats_raw)
    except Exception as e:
        raise ValueError(f"const_repeat: repeats must be an integer, got {repeats_raw!r}.") from e
    if repeats <= 0:
        raise ValueError(f"const_repeat: repeats must be > 0, got {repeats}.")

    if "var" in dsl:
        var_name = str(dsl.get("var", _default_var_name(ctx)))
        v = next((x for x in ctx.pack.vars if x.name == var_name), None)
        if v is None:
            raise ValueError(f"const_repeat: unknown variable: {var_name!r}")
        return RepeatConstantExpr(
            name=dsl.get("name", "const_repeat"),
            value=np.asarray(dsl["value"], float),
            repeats=repeats,
            vars=[v],
        )

    return RepeatConstantExpr(
        name=dsl.get("name", "const_repeat"),
        value=np.asarray(dsl["value"], float),
        repeats=repeats,
    )


def build_get_state(ctx, dsl):
    key_dsl = dsl["key"]
    k = int(key_dsl.get("k", 0))
    owner = OwnerKey(key_dsl["owner_type"], key_dsl["owner_name"])
    dtype = canonical_dtype_name(str(key_dsl["dtype"]))
    field = canonical_field_name(str(key_dsl["field"]))
    frame = key_dsl.get("frame", None)
    rel_frame = key_dsl.get("rel_frame", None)

    if dtype == DTYPE_KINEMATICS:
        if frame is None:
            frame = DEFAULT_FRAME
        elif frame != DEFAULT_FRAME:
            raise ValueError(f"get_state: currently only frame='{DEFAULT_FRAME}' is supported (got {frame!r})")

    key_value = StateKey(k=k, owner=owner, dtype=dtype, field=field, frame=frame, rel_frame=rel_frame)

    jac_entries: list[dict] = []
    jac_dsl = dsl.get("jac", None)
    if jac_dsl is not None:
        if not isinstance(jac_dsl, dict):
            raise ValueError("get_state: jac must be a dict when provided.")
        jac_entries.append(dict(jac_dsl))

    jacs_dsl = dsl.get("jacs", None)
    if jacs_dsl is not None:
        if not isinstance(jacs_dsl, list):
            raise ValueError("get_state: jacs must be a list[dict] when provided.")
        for i, item in enumerate(jacs_dsl):
            if not isinstance(item, dict):
                raise ValueError(f"get_state: jacs[{i}] must be a dict, got {type(item).__name__}.")
            jac_entries.append(dict(item))

    if len(jac_entries) == 0:
        jac_entries = [{}]

    vars_out = []
    key_jacs = []
    seen_var_names = set()
    for jac_entry in jac_entries:
        jac_var = jac_entry.get("var", None)
        if jac_var is None:
            jac_var = _default_var_name(ctx)
        jac_var = str(jac_var)
        if jac_var in seen_var_names:
            raise ValueError(f"get_state: duplicate jac var {jac_var!r}.")
        seen_var_names.add(jac_var)

        v = next((x for x in ctx.pack.vars if x.name == jac_var), None)
        if v is None:
            raise ValueError(f"get_state: unknown jac variable: {jac_var!r}")

        jac_field_name_raw = jac_entry.get("field", jac_field(field, var=jac_var))
        jac_field_name = canonical_field_name(str(jac_field_name_raw))
        key_jac = StateKey(k=k, owner=owner, dtype=dtype, field=jac_field_name, frame=frame, rel_frame=rel_frame)
        vars_out.append(v)
        key_jacs.append(key_jac)

    return GetStateExpr(
        name=dsl.get("name", "get_state"),
        vars=vars_out,
        key_value=key_value,
        key_jacs=key_jacs,
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

    deriv_order_raw = dsl.get("derivative_order", dsl.get("deriv_order", 0))
    try:
        deriv_order = int(deriv_order_raw)
    except Exception as e:
        raise ValueError(
            "get_traj_var: derivative_order must be an integer, "
            f"got {deriv_order_raw!r}."
        ) from e
    if deriv_order < 0:
        raise ValueError(f"get_traj_var: derivative_order must be >= 0, got {deriv_order}.")

    deriv_wrt = str(dsl.get("derivative_wrt", dsl.get("wrt", "u"))).strip().lower()
    if deriv_wrt == "":
        deriv_wrt = "u"

    max_deriv_order_raw = dsl.get(
        "max_derivative_order",
        dsl.get("derivative_order_max", dsl.get("max_deriv_order", None)),
    )
    if max_deriv_order_raw is None:
        max_deriv_order = None
    else:
        try:
            max_deriv_order = int(max_deriv_order_raw)
        except Exception as e:
            raise ValueError(
                "get_traj_var: max_derivative_order must be an integer, "
                f"got {max_deriv_order_raw!r}."
            ) from e
        if max_deriv_order < 0:
            raise ValueError(
                "get_traj_var: max_derivative_order must be >= 0, "
                f"got {max_deriv_order}."
            )

    resolve_traj = getattr(ctx, "resolve_trajectory_map", None)

    if max_deriv_order is not None:
        if deriv_order != 0:
            raise ValueError(
                "get_traj_var: derivative_order and max_derivative_order cannot be used together. "
                "Use either derivative_order=<r> or max_derivative_order=<N>."
            )
        if max_deriv_order == 0 and callable(resolve_traj):
            trajectories = [resolve_traj(traj_dsl, default_q_dim=default_q_dim)]
        else:
            trajectories = build_trajectory_maps_with_derivatives(
                traj_dsl,
                max_derivative_order=max_deriv_order,
                derivative_wrt=deriv_wrt,
                default_steps=default_steps_from_time(getattr(ctx, "time", None)),
                default_q_dim=default_q_dim,
                default_dt=default_dt_from_time(getattr(ctx, "time", None)),
            )
        if any(traj.p_dim != v.dim() for traj in trajectories):
            got = [traj.p_dim for traj in trajectories]
            raise ValueError(
                "get_traj_var: variable dimension mismatch with trajectory parameter dimension. "
                f"var {var_name!r} dim={v.dim()}, trajectory p_dim(s)={got}."
            )
    else:
        if deriv_order == 0 and callable(resolve_traj):
            traj = resolve_traj(traj_dsl, default_q_dim=default_q_dim)
        elif deriv_order == 0:
            traj = build_trajectory_map(
                traj_dsl,
                default_steps=default_steps_from_time(getattr(ctx, "time", None)),
                default_q_dim=default_q_dim,
            )
        else:
            traj = build_trajectory_map_with_derivative(
                traj_dsl,
                derivative_order=deriv_order,
                derivative_wrt=deriv_wrt,
                default_steps=default_steps_from_time(getattr(ctx, "time", None)),
                default_q_dim=default_q_dim,
                default_dt=default_dt_from_time(getattr(ctx, "time", None)),
            )
        if traj.p_dim != v.dim():
            raise ValueError(
                "get_traj_var: variable dimension mismatch with trajectory parameter dimension. "
                f"var {var_name!r} dim={v.dim()}, trajectory p_dim={traj.p_dim}."
            )

    k = dsl.get("k", None)
    if k is None:
        key = dsl.get("key", None)
        if isinstance(key, dict) and "k" in key:
            k = key.get("k", None)
    k_i = None if k is None else int(k)

    if max_deriv_order is not None:
        return TrajectoryVarDerivativesExpr(
            name=dsl.get("name", "get_traj_var"),
            vars=[v],
            trajectories=trajectories,
            k=k_i,
        )

    return TrajectoryVarExpr(name=dsl.get("name", "get_traj_var"), vars=[v], trajectory=traj, k=k_i)


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
        inner_k["k"] = k
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

    wrt = str(dsl.get("wrt", dsl.get("derivative_wrt", "index"))).strip().lower()
    divide_by_dt = bool(dsl.get("divide_by_dt", False))

    use_time_dt = False
    if wrt in ("time", "t"):
        use_time_dt = True
    elif wrt in ("index", "step", "k", "sample"):
        use_time_dt = False
    else:
        raise ValueError(
            "time_diff: wrt must be one of 'index', 'step', 'k', 'sample', 'time', 't'. "
            f"Got {wrt!r}."
        )
    if divide_by_dt:
        use_time_dt = True

    dt_raw = dsl.get("dt", None)
    dt = None
    if dt_raw is not None:
        try:
            dt = float(dt_raw)
        except Exception as e:
            raise ValueError(f"time_diff: dt must be a float, got {dt_raw!r}.") from e
        if dt <= 0.0:
            raise ValueError(f"time_diff: dt must be > 0, got {dt}.")

    scale_raw = dsl.get("scale", 1.0)
    try:
        scale = float(scale_raw)
    except Exception as e:
        raise ValueError(f"time_diff: scale must be a float, got {scale_raw!r}.") from e

    return TimeDiffExpr(
        name=dsl.get("name", "time_diff"),
        base=base,
        segment_dim=segment_dim,
        scale=scale,
        use_time_dt=use_time_dt,
        dt=dt,
    )
