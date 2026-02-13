from __future__ import annotations

from collections.abc import Mapping

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
    pick_trajectory_value,
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


def _resolve_positive_int(value, *, where: str) -> int:
    try:
        out = int(value)
    except Exception as e:
        raise ValueError(f"{where}: expected positive integer, got {value!r}.") from e
    if out <= 0:
        raise ValueError(f"{where}: expected positive integer, got {out}.")
    return out


def _time_steps_from_ctx(ctx) -> int | None:
    time = getattr(ctx, "time", None)
    if time is None or not hasattr(time, "N"):
        return None
    try:
        steps = int(time.N) + 1
    except Exception:
        return None
    return steps if steps > 0 else None


def _resolve_time_index(raw, *, ctx, where: str, allow_none: bool = False) -> int | None:
    if raw is None:
        return None if allow_none else 0

    steps = _time_steps_from_ctx(ctx)

    if isinstance(raw, str):
        token = raw.strip().lower()
        if token == "":
            raise ValueError(f"{where}: k must be non-empty.")
        if token in ("last", "end", "final"):
            if steps is None:
                raise ValueError(f"{where}: k={raw!r} requires time.N in DSL.")
            return int(steps - 1)
        if token in ("first", "start"):
            return 0
        if token.startswith("last-") or token.startswith("end-") or token.startswith("final-"):
            if steps is None:
                raise ValueError(f"{where}: k={raw!r} requires time.N in DSL.")
            tail = token.split("-", 1)[1].strip()
            if tail == "":
                raise ValueError(f"{where}: invalid k specifier {raw!r}.")
            try:
                offset = int(tail)
            except Exception as e:
                raise ValueError(f"{where}: invalid k offset in {raw!r}.") from e
            if offset < 0:
                raise ValueError(f"{where}: k offset must be >= 0, got {offset}.")
            k = int(steps - 1 - offset)
        else:
            try:
                k = int(raw)
            except Exception as e:
                raise ValueError(
                    f"{where}: k must be int or one of 'last', 'last-<n>', 'first'. Got {raw!r}."
                ) from e
    else:
        try:
            k = int(raw)
        except Exception as e:
            raise ValueError(
                f"{where}: k must be int or one of 'last', 'last-<n>', 'first'. Got {raw!r}."
            ) from e

    if k < 0:
        raise ValueError(f"{where}: k must be >= 0, got {k}.")
    if steps is not None and k >= steps:
        raise ValueError(f"{where}: k={k} out of range for time steps 0..{steps - 1}.")
    return int(k)


def _parse_fill_value(raw: object, *, where: str) -> float | None:
    if not isinstance(raw, Mapping):
        return None
    if "fill" not in raw:
        raise ValueError(f"{where}: dict value must contain key 'fill'.")
    try:
        return float(raw["fill"])
    except Exception as e:
        raise ValueError(f"{where}: value.fill must be numeric, got {raw['fill']!r}.") from e


def _validate_vector_dim(value: np.ndarray, *, dim: int, where: str) -> np.ndarray:
    vec = np.asarray(value, dtype=float).reshape(-1)
    dim_i = int(dim)
    if dim_i <= 0:
        raise ValueError(f"{where}: dim must be > 0, got {dim_i}.")
    if vec.size == dim_i:
        return vec.copy()
    if vec.size == 1 and dim_i > 1:
        raise ValueError(
            f"{where}: scalar value is not broadcast implicitly. "
            "Use `value = { fill = <value> }` to fill all elements."
        )
    raise ValueError(f"{where}: value size {vec.size} is not compatible with dim {dim_i}.")


def _infer_const_repeat_segment_dim(ctx, dsl, *, repeats: int) -> int | None:
    seg_dim_raw = dsl.get("segment_dim", dsl.get("dim", None))
    if seg_dim_raw is not None:
        return _resolve_positive_int(seg_dim_raw, where="const_repeat.segment_dim")

    traj_dsl = dsl.get("trajectory", None)
    if traj_dsl is None:
        root_dsl = getattr(ctx, "root_dsl", None)
        if isinstance(root_dsl, dict):
            traj_dsl = root_dsl.get("trajectory", None)
    if not isinstance(traj_dsl, dict):
        return None

    resolve_traj = getattr(ctx, "resolve_trajectory_map", None)
    if callable(resolve_traj):
        try:
            traj = resolve_traj(traj_dsl, default_q_dim=None)
            if int(traj.steps) == int(repeats):
                return int(traj.q_dim)
        except Exception:
            pass

    typ = str(traj_dsl.get("type", "")).strip().lower()
    if typ == "":
        return None
    steps_raw = pick_trajectory_value(traj_dsl, section=typ, key="steps")
    q_dim_raw = pick_trajectory_value(traj_dsl, section=typ, key="q_dim")
    if steps_raw is None or q_dim_raw is None:
        return None
    try:
        steps = int(steps_raw)
        q_dim = int(q_dim_raw)
    except Exception:
        return None
    if steps <= 0 or q_dim <= 0:
        return None
    if int(steps) != int(repeats):
        return None
    return int(q_dim)


def _infer_const_fill_dim_from_trajectory(
    ctx,
    *,
    var_name: str,
    var_dim: int,
) -> int | None:
    root_dsl = getattr(ctx, "root_dsl", None)
    if not isinstance(root_dsl, dict):
        return None
    traj_dsl = root_dsl.get("trajectory", None)
    if not isinstance(traj_dsl, dict):
        return None

    traj_var_raw = traj_dsl.get("var", None)
    if traj_var_raw is not None and str(traj_var_raw).strip() not in ("", str(var_name)):
        return None

    resolve_traj = getattr(ctx, "resolve_trajectory_map", None)
    if callable(resolve_traj):
        try:
            default_q_dim = None
            if str(traj_dsl.get("type", "")).strip().lower() == "bspline":
                default_q_dim = infer_bspline_q_dim_from_var(traj_dsl, var_dim=int(var_dim))
            traj = resolve_traj(traj_dsl, default_q_dim=default_q_dim)
            return int(traj.q_dim)
        except Exception:
            pass

    typ = str(traj_dsl.get("type", "")).strip().lower()
    if typ == "":
        return None
    q_dim_raw = pick_trajectory_value(traj_dsl, section=typ, key="q_dim")
    if q_dim_raw is not None:
        try:
            q_dim = int(q_dim_raw)
            if q_dim > 0:
                return q_dim
        except Exception:
            return None
    if typ == "bspline":
        return infer_bspline_q_dim_from_var(traj_dsl, var_dim=int(var_dim))
    return None


def build_const(ctx, dsl):
    value_raw = dsl["value"]
    fill = _parse_fill_value(value_raw, where="const")
    value = np.asarray(value_raw, float).reshape(-1) if fill is None else None

    var_name = None
    v = None
    if "var" in dsl:
        var_name = str(dsl.get("var", _default_var_name(ctx)))
        v = next((x for x in ctx.pack.vars if x.name == var_name), None)
        if v is None:
            raise ValueError(f"const: unknown variable: {var_name!r}")

    dim_raw = dsl.get("dim", None)
    dim = None if dim_raw is None else _resolve_positive_int(dim_raw, where="const.dim")

    if fill is not None:
        if dim is None:
            if v is None:
                raise ValueError("const: value.fill requires dim or var.")
            inferred = _infer_const_fill_dim_from_trajectory(
                ctx,
                var_name=str(var_name),
                var_dim=int(v.dim()),
            )
            dim = int(v.dim()) if inferred is None else int(inferred)
        value = np.full((dim,), float(fill), dtype=float)
    else:
        if dim is not None:
            value = _validate_vector_dim(np.asarray(value, dtype=float), dim=dim, where="const")
        else:
            value = np.asarray(value, dtype=float).reshape(-1)
            if value.size == 1 and v is not None and int(v.dim()) > 1:
                raise ValueError(
                    "const: scalar value with multi-dimensional var is ambiguous. "
                    "Use `value = { fill = <value> }` for all-equal vectors, "
                    "or set const.dim explicitly."
                )

    if v is not None:
        return ConstantExpr(name=dsl.get("name", "const"), vars=[v], value=value)
    return ConstantExpr(name=dsl.get("name", "const"), value=value)


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

    value_raw = dsl["value"]
    fill = _parse_fill_value(value_raw, where="const_repeat")
    seg_dim = _infer_const_repeat_segment_dim(ctx, dsl, repeats=repeats)

    if fill is not None:
        if seg_dim is None:
            raise ValueError(
                "const_repeat: value.fill requires segment_dim/dim, "
                "or trajectory with inferable q_dim."
            )
        value = np.full((seg_dim,), float(fill), dtype=float)
    else:
        value = np.asarray(value_raw, dtype=float).reshape(-1)
        if seg_dim is not None:
            value = _validate_vector_dim(value, dim=seg_dim, where="const_repeat")

    if "var" in dsl:
        var_name = str(dsl.get("var", _default_var_name(ctx)))
        v = next((x for x in ctx.pack.vars if x.name == var_name), None)
        if v is None:
            raise ValueError(f"const_repeat: unknown variable: {var_name!r}")
        return RepeatConstantExpr(
            name=dsl.get("name", "const_repeat"),
            value=value,
            repeats=repeats,
            vars=[v],
        )

    return RepeatConstantExpr(
        name=dsl.get("name", "const_repeat"),
        value=value,
        repeats=repeats,
    )


def build_get_state(ctx, dsl):
    key_dsl = dsl["key"]
    k = int(_resolve_time_index(key_dsl.get("k", 0), ctx=ctx, where="get_state.key.k"))
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
    k_i = _resolve_time_index(k, ctx=ctx, where="get_var.k", allow_none=True)

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
    k_i = _resolve_time_index(k, ctx=ctx, where="get_traj_var.k", allow_none=True)

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
    k0 = int(_resolve_time_index(r["k0"], ctx=ctx, where="stack.range.k0"))
    k1 = int(_resolve_time_index(r["k1"], ctx=ctx, where="stack.range.k1"))
    if k1 < k0:
        raise ValueError(f"stack.range: expected k0 <= k1, got k0={k0}, k1={k1}.")
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
