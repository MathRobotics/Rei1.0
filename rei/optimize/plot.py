from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.state_cache import OwnerKey, StateKey
from ..core.state_schema import canonical_dtype_name, canonical_field_name
from .runtime import NLSRuntime

Array = np.ndarray
_PLOT_TYPE_STATE_TRAJ = "state_traj"
_PLOT_TYPES_TRAJ_DERIVATIVE = frozenset(
    ("traj_derivative", "trajectory_derivative", "joint_traj_derivative")
)


@dataclass(frozen=True)
class TermAttrPlotSeries:
    term_index: int
    term_name: str
    name: str
    owner_type: str
    owner_name: str
    dtype: str
    field: str
    frame: str | None
    rel_frame: str | None
    ks: tuple[int, ...]
    x: Array
    y: Array
    x_axis: str
    component_labels: tuple[str, ...] = ()

    def line_label(self, component: int) -> str:
        i = int(component)
        cols = int(self.y.shape[1])
        if i < 0 or i >= cols:
            raise IndexError(f"line_label: component index out of range: {i}. Expected 0..{cols - 1}.")

        if cols == 1:
            return str(self.name)

        if i < len(self.component_labels):
            label = str(self.component_labels[i]).strip()
            if label != "":
                return label
        return f"{self.name}[{i}]"


def _iter_expr_children(expr: Any) -> Iterable[Any]:
    for attr in ("a", "b", "base", "inner"):
        child = getattr(expr, attr, None)
        if child is not None and hasattr(child, "eval"):
            yield child

    parts = getattr(expr, "parts", None)
    if isinstance(parts, (list, tuple)):
        for child in parts:
            if child is not None and hasattr(child, "eval"):
                yield child


def _walk_expr(expr: Any) -> Iterable[Any]:
    stack = [expr]
    while stack:
        cur = stack.pop()
        yield cur
        children = list(_iter_expr_children(cur))
        stack.extend(reversed(children))


def _first_state_key_in_expr(expr: Any) -> StateKey | None:
    for node in _walk_expr(expr):
        key = getattr(node, "key_value", None)
        if isinstance(key, StateKey):
            return key
    return None


def _normalize_term_indices(
    runtime: NLSRuntime,
    *,
    term_indices: Iterable[int] | None,
) -> list[int]:
    n_terms = len(runtime.problem.terms)
    if term_indices is None:
        return list(range(n_terms))

    out: list[int] = []
    seen: set[int] = set()
    for idx_raw in term_indices:
        idx = int(idx_raw)
        if idx < 0 or idx >= n_terms:
            raise IndexError(
                "collect_plot_series_from_term_attrs: term index out of range. "
                f"Got {idx}, expected 0..{n_terms - 1}."
            )
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def _coerce_plot_specs(raw: Any, *, where: str, strict: bool) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        return [dict(raw)]
    if isinstance(raw, list):
        out: list[dict[str, Any]] = []
        for i, item in enumerate(raw):
            if not isinstance(item, Mapping):
                if strict:
                    raise ValueError(f"{where}[{i}] must be a mapping.")
                continue
            out.append(dict(item))
        return out

    if strict:
        raise ValueError(f"{where} must be a mapping or list[mapping].")
    return []


def _time_steps(runtime: NLSRuntime) -> int | None:
    time = runtime.time
    if time is None or not hasattr(time, "N"):
        return None
    try:
        steps = int(time.N) + 1
    except Exception:
        return None
    if steps <= 0:
        return None
    return steps


def _parse_time_index(raw: Any, *, steps: int | None, where: str) -> int:
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token == "":
            raise ValueError(f"{where}: k must be non-empty.")
        if token in ("first", "start"):
            return 0
        if token in ("last", "end", "final"):
            if steps is None:
                raise ValueError(f"{where}: k={raw!r} requires runtime.time.N.")
            return int(steps - 1)
        if token.startswith("last-") or token.startswith("end-") or token.startswith("final-"):
            if steps is None:
                raise ValueError(f"{where}: k={raw!r} requires runtime.time.N.")
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
                k = int(token)
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


def _resolve_ks(
    spec: Mapping[str, Any],
    *,
    runtime: NLSRuntime,
    fallback_key: StateKey | None,
    where: str,
) -> tuple[int, ...]:
    steps = _time_steps(runtime)

    if "ks" in spec:
        if "k0" in spec or "k1" in spec:
            raise ValueError(f"{where}: use either `ks` or (`k0`, `k1`), not both.")
        raw_ks = spec.get("ks", None)
        if not isinstance(raw_ks, Sequence) or isinstance(raw_ks, (str, bytes)):
            raise ValueError(f"{where}: ks must be a sequence of indices.")
        return tuple(
            _parse_time_index(k_raw, steps=steps, where=f"{where}.ks[{i}]")
            for i, k_raw in enumerate(raw_ks)
        )

    k0 = _parse_time_index(spec.get("k0", 0), steps=steps, where=f"{where}.k0")
    if "k1" in spec:
        k1_raw = spec["k1"]
    elif steps is not None:
        k1_raw = "last"
    elif fallback_key is not None:
        k1_raw = int(fallback_key.k)
    else:
        k1_raw = 0
    k1 = _parse_time_index(k1_raw, steps=steps, where=f"{where}.k1")
    if k1 < k0:
        raise ValueError(f"{where}: expected k0 <= k1, got k0={k0}, k1={k1}.")
    return tuple(range(k0, k1 + 1))


def _resolve_state_route(
    spec: Mapping[str, Any],
    *,
    fallback_key: StateKey | None,
    where: str,
) -> StateKey:
    key_raw = spec.get("key", None)
    if key_raw is None:
        key_dsl: dict[str, Any] = {}
    elif isinstance(key_raw, Mapping):
        key_dsl = dict(key_raw)
    else:
        raise ValueError(f"{where}.key must be a mapping when provided.")

    def pick(name: str, *, required: bool) -> Any:
        if name in spec:
            return spec[name]
        if name in key_dsl:
            return key_dsl[name]
        if fallback_key is not None:
            if name == "owner_type":
                return fallback_key.owner.owner_type
            if name == "owner_name":
                return fallback_key.owner.owner_name
            if name == "dtype":
                return fallback_key.dtype
            if name == "field":
                return fallback_key.field
            if name == "frame":
                return fallback_key.frame
            if name == "rel_frame":
                return fallback_key.rel_frame
        if required:
            raise ValueError(
                f"{where}: missing required state key field {name!r}. "
                "Specify it in attrs.plot or include a get_state node in the term for inference."
            )
        return None

    owner_type = str(pick("owner_type", required=True)).strip()
    owner_name = str(pick("owner_name", required=True)).strip()
    dtype = canonical_dtype_name(str(pick("dtype", required=True)).strip())
    field = canonical_field_name(str(pick("field", required=True)).strip())
    frame_raw = pick("frame", required=False)
    rel_frame_raw = pick("rel_frame", required=False)

    if owner_type == "" or owner_name == "":
        raise ValueError(f"{where}: owner_type and owner_name must be non-empty.")

    frame = None if frame_raw is None else str(frame_raw)
    rel_frame = None if rel_frame_raw is None else str(rel_frame_raw)
    return StateKey(
        k=0,
        owner=OwnerKey(owner_type=owner_type, owner_name=owner_name),
        dtype=dtype,
        field=field,
        frame=frame,
        rel_frame=rel_frame,
    )


def _resolve_expected_dim(spec: Mapping[str, Any], *, where: str) -> int | None:
    raw = spec.get("expected_dim", None)
    if raw is None:
        return None
    try:
        dim = int(raw)
    except Exception as e:
        raise ValueError(f"{where}.expected_dim must be an integer, got {raw!r}.") from e
    if dim < 0:
        raise ValueError(f"{where}.expected_dim must be >= 0, got {dim}.")
    return dim


def _resolve_component_labels(spec: Mapping[str, Any], *, dim: int, where: str) -> tuple[str, ...]:
    raw = spec.get("components", spec.get("component_labels", None))
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{where}: components must be a sequence of labels.")
    labels = tuple(str(item) for item in raw)
    if len(labels) != int(dim):
        raise ValueError(
            f"{where}: components length mismatch. Expected {int(dim)}, got {len(labels)}."
        )
    return labels


def _resolve_x_axis(spec: Mapping[str, Any], *, runtime: NLSRuntime, where: str) -> str:
    raw = str(spec.get("x_axis", spec.get("x", "auto"))).strip().lower()
    has_time = runtime.time is not None and hasattr(runtime.time, "t")

    if raw in ("", "auto"):
        return "time" if has_time else "index"
    if raw in ("time", "t", "seconds", "sec", "s"):
        return "time"
    if raw in ("index", "k", "step"):
        return "index"
    raise ValueError(
        f"{where}: x_axis must be one of 'auto', 'time', 'index'. Got {raw!r}."
    )


def _build_x_values(*, ks: Sequence[int], runtime: NLSRuntime, x_axis: str) -> tuple[Array, str]:
    if x_axis == "time":
        time = runtime.time
        if time is not None and hasattr(time, "t"):
            out = np.asarray([float(time.t(int(k))) for k in ks], dtype=float)
            return out, "time"
    out = np.asarray([float(int(k)) for k in ks], dtype=float)
    return out, "index"


def _term_name(expr: Any) -> str:
    name = getattr(expr, "name", None)
    if isinstance(name, str) and name:
        return name
    return expr.__class__.__name__


def collect_plot_series_from_term_attrs(
    runtime: NLSRuntime,
    *,
    term_indices: Iterable[int] | None = None,
    strict: bool = True,
) -> list[TermAttrPlotSeries]:
    """Collect state trajectories declared by `term.attrs.plot` metadata.

    Supported spec type in this collector:
      - `type = "state_traj"`

    Notes:
      - `type = "traj_derivative"` and aliases are ignored here.
        Use `collect_trajectory_derivative_plot_series_from_term_attrs`
        or `collect_plot_series_from_compiled_term_attrs` for those.
    """

    out: list[TermAttrPlotSeries] = []
    indices = _normalize_term_indices(runtime, term_indices=term_indices)

    for idx in indices:
        expr, _cost = runtime.problem.terms[idx]
        term_name = _term_name(expr)
        attrs = runtime.problem.term_attrs_at(idx)
        specs = _coerce_plot_specs(
            attrs.get("plot", None),
            where=f"term[{idx}].attrs.plot",
            strict=strict,
        )
        if len(specs) == 0:
            continue

        fallback_key = _first_state_key_in_expr(expr)

        for j, spec in enumerate(specs):
            where = f"term[{idx}].attrs.plot[{j}]"
            typ = str(spec.get("type", spec.get("kind", _PLOT_TYPE_STATE_TRAJ))).strip().lower()
            if typ in _PLOT_TYPES_TRAJ_DERIVATIVE:
                continue
            if typ != _PLOT_TYPE_STATE_TRAJ:
                if strict:
                    raise ValueError(
                        f"{where}: unsupported plot type {typ!r}. "
                        "Supported types are 'state_traj' (this collector) "
                        "and trajectory-derivative plot types for compiled collectors."
                    )
                continue

            route = _resolve_state_route(spec, fallback_key=fallback_key, where=where)
            ks = _resolve_ks(spec, runtime=runtime, fallback_key=fallback_key, where=where)
            if len(ks) == 0:
                if strict:
                    raise ValueError(f"{where}: resolved ks is empty.")
                continue

            y = runtime.collect_state_traj(
                owner_type=route.owner.owner_type,
                owner_name=route.owner.owner_name,
                dtype=route.dtype,
                field=route.field,
                ks=ks,
                frame=route.frame,
                rel_frame=route.rel_frame,
                expected_dim=_resolve_expected_dim(spec, where=where),
            )
            y = np.asarray(y, dtype=float)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if y.ndim != 2:
                raise ValueError(f"{where}: expected 2D trajectory array, got shape {y.shape}.")

            x_axis = _resolve_x_axis(spec, runtime=runtime, where=where)
            x, x_axis_resolved = _build_x_values(ks=ks, runtime=runtime, x_axis=x_axis)
            if int(x.size) != int(y.shape[0]):
                raise RuntimeError(
                    f"{where}: internal x/y length mismatch. x={x.size}, y_rows={y.shape[0]}."
                )

            name_raw = spec.get("name", spec.get("label", None))
            if name_raw is None:
                name = f"{term_name}:{route.field}"
            else:
                name = str(name_raw).strip()
                if name == "":
                    name = f"{term_name}:{route.field}"

            out.append(
                TermAttrPlotSeries(
                    term_index=int(idx),
                    term_name=term_name,
                    name=name,
                    owner_type=route.owner.owner_type,
                    owner_name=route.owner.owner_name,
                    dtype=route.dtype,
                    field=route.field,
                    frame=route.frame,
                    rel_frame=route.rel_frame,
                    ks=tuple(int(k) for k in ks),
                    x=x.copy(),
                    y=y.copy(),
                    x_axis=x_axis_resolved,
                    component_labels=_resolve_component_labels(
                        spec,
                        dim=int(y.shape[1]),
                        where=where,
                    ),
                )
            )

    return out


def _parse_derivative_order(raw: Any, *, where: str, strict: bool) -> int | None:
    if raw is None:
        if strict:
            raise ValueError(
                f"{where}: missing derivative order. Use `derivative_order` or `order`."
            )
        return None

    try:
        order = int(raw)
    except Exception as e:
        if strict:
            raise ValueError(f"{where}: derivative order must be an integer, got {raw!r}.") from e
        return None

    if order < 0:
        if strict:
            raise ValueError(f"{where}: derivative order must be >= 0, got {order}.")
        return None

    return int(order)


def _collect_derivative_plot_requests_from_term_attrs(
    runtime: NLSRuntime,
    *,
    term_indices: Iterable[int] | None = None,
    strict: bool = True,
) -> tuple[list[int], dict[int, str]]:
    indices = _normalize_term_indices(runtime, term_indices=term_indices)

    orders: list[int] = []
    seen_orders: set[int] = set()
    names: dict[int, str] = {}
    for idx in indices:
        attrs = runtime.problem.term_attrs_at(idx)
        specs = _coerce_plot_specs(
            attrs.get("plot", None),
            where=f"term[{idx}].attrs.plot",
            strict=strict,
        )
        if len(specs) == 0:
            continue

        for j, spec in enumerate(specs):
            where = f"term[{idx}].attrs.plot[{j}]"
            typ = str(spec.get("type", spec.get("kind", _PLOT_TYPE_STATE_TRAJ))).strip().lower()
            if typ not in _PLOT_TYPES_TRAJ_DERIVATIVE:
                continue

            order = _parse_derivative_order(
                spec.get("derivative_order", spec.get("order", None)),
                where=where,
                strict=strict,
            )
            if order is None:
                continue

            if order not in seen_orders:
                seen_orders.add(order)
                orders.append(int(order))

            if "name" not in spec:
                continue

            name = str(spec.get("name", "")).strip()
            if name == "":
                if strict:
                    raise ValueError(f"{where}: name must be non-empty when provided.")
                continue

            existing = names.get(int(order), None)
            if existing is not None and existing != name and strict:
                raise ValueError(
                    f"{where}: conflicting name for derivative order={order}. "
                    f"Existing={existing!r}, got={name!r}."
                )
            names[int(order)] = name

    return orders, names


def _default_joint_series_name(order: int) -> str:
    if order == 0:
        return "joint_q"
    if order == 1:
        return "joint_qdot"
    if order == 2:
        return "joint_qddot"
    return f"joint_q_d{order}"


def _default_joint_series_field(order: int) -> str:
    if order == 0:
        return "q"
    if order == 1:
        return "qdot"
    if order == 2:
        return "qddot"
    return f"q_d{order}"


def _resolve_runtime_pack_var(
    runtime: NLSRuntime,
    *,
    var_name: str,
) -> Array:
    pack = getattr(runtime, "pack", None)
    if pack is None:
        raise ValueError("collect_trajectory_derivative_plot_series: runtime.pack is None.")

    vars_list = getattr(pack, "vars", None)
    if vars_list is None:
        raise ValueError("collect_trajectory_derivative_plot_series: runtime.pack.vars is missing.")

    var_name_norm = str(var_name)
    for var in vars_list:
        if str(getattr(var, "name", "")) != var_name_norm:
            continue
        return np.asarray(getattr(var, "x"), dtype=float).reshape(-1).copy()

    raise ValueError(
        "collect_trajectory_derivative_plot_series: "
        f"variable {var_name_norm!r} was not found in runtime.pack.vars."
    )


def _resolve_traj_x_values(
    *,
    runtime: NLSRuntime,
    steps: int,
    dt: float | None,
) -> tuple[Array, str]:
    time = runtime.time
    if time is not None and hasattr(time, "t"):
        try:
            x_time = np.asarray([float(time.t(int(k))) for k in range(int(steps))], dtype=float)
            if x_time.size <= 1:
                return x_time, "time"
            dx = np.diff(x_time)
            if np.any(np.abs(dx) > 0.0):
                return x_time, "time"
        except Exception:
            pass

    if dt is not None:
        dt_f = float(dt)
        if dt_f > 0.0:
            return dt_f * np.arange(int(steps), dtype=float), "time"

    return np.arange(int(steps), dtype=float), "index"


def collect_trajectory_derivative_plot_series(
    compiled: Any,
    *,
    derivative_orders: Iterable[int] = (1, 2),
    names: Mapping[int, str] | None = None,
    p: Array | None = None,
    strict: bool = False,
) -> list[TermAttrPlotSeries]:
    """Collect trajectory derivative plot series from a compiled trajectory problem.

    The `compiled` object is expected to expose:
      - `runtime`
      - `trajectory_derivative_maps`
      - `p_var` (used when `p` is not given)
      - optional `dt`
    """

    runtime = getattr(compiled, "runtime", None)
    if not isinstance(runtime, NLSRuntime):
        raise TypeError(
            "collect_trajectory_derivative_plot_series: "
            "compiled.runtime must be an NLSRuntime."
        )

    maps_raw = getattr(compiled, "trajectory_derivative_maps", None)
    if not isinstance(maps_raw, Mapping):
        raise TypeError(
            "collect_trajectory_derivative_plot_series: "
            "compiled.trajectory_derivative_maps must be a mapping."
        )
    maps = {int(k): v for k, v in maps_raw.items()}

    order_list: list[int] = []
    seen_orders: set[int] = set()
    for order_raw in derivative_orders:
        order = int(order_raw)
        if order < 0:
            raise ValueError(
                "collect_trajectory_derivative_plot_series: "
                f"derivative order must be >= 0, got {order}."
            )
        if order in seen_orders:
            continue
        seen_orders.add(order)
        order_list.append(order)

    if p is None:
        p_var = str(getattr(compiled, "p_var", "")).strip()
        if p_var == "":
            raise ValueError(
                "collect_trajectory_derivative_plot_series: "
                "compiled.p_var must be non-empty when `p` is omitted."
            )
        p_vec = _resolve_runtime_pack_var(runtime, var_name=p_var)
    else:
        p_vec = np.asarray(p, dtype=float).reshape(-1).copy()

    dt_raw = getattr(compiled, "dt", None)
    dt = None if dt_raw is None else float(dt_raw)
    name_overrides: dict[int, str] = {}
    if names is not None:
        name_overrides = {int(k): str(v) for k, v in names.items()}

    out: list[TermAttrPlotSeries] = []
    for order in order_list:
        traj = maps.get(int(order), None)
        if traj is None:
            if strict:
                raise ValueError(
                    "collect_trajectory_derivative_plot_series: "
                    f"trajectory derivative map for order={order} was not found."
                )
            continue

        A = np.asarray(getattr(traj, "A"), dtype=float)
        b = np.asarray(getattr(traj, "b"), dtype=float).reshape(-1)
        steps = int(getattr(traj, "steps"))
        q_dim = int(getattr(traj, "q_dim"))
        if A.ndim != 2:
            raise ValueError(
                "collect_trajectory_derivative_plot_series: "
                f"trajectory map A must be 2D, got shape {A.shape} for order={order}."
            )
        if int(A.shape[1]) != int(p_vec.size):
            raise ValueError(
                "collect_trajectory_derivative_plot_series: "
                "parameter size mismatch while evaluating trajectory map. "
                f"order={order}, A shape={A.shape}, p size={p_vec.size}."
            )

        y = (A @ p_vec + b).reshape(steps, q_dim)
        x, x_axis = _resolve_traj_x_values(runtime=runtime, steps=steps, dt=dt)

        name_raw = name_overrides.get(order, _default_joint_series_name(order))
        name = str(name_raw).strip()
        if name == "":
            name = _default_joint_series_name(order)

        out.append(
            TermAttrPlotSeries(
                term_index=-(int(order) + 1),
                term_name="trajectory",
                name=name,
                owner_type="total_joint",
                owner_name="trajectory",
                dtype="coord",
                field=_default_joint_series_field(order),
                frame=None,
                rel_frame=None,
                ks=tuple(range(steps)),
                x=x.copy(),
                y=y.copy(),
                x_axis=x_axis,
                component_labels=tuple(f"{name}[{i}]" for i in range(int(q_dim))),
            )
        )

    return out


def collect_trajectory_derivative_plot_series_from_term_attrs(
    compiled: Any,
    *,
    term_indices: Iterable[int] | None = None,
    p: Array | None = None,
    strict: bool = True,
) -> list[TermAttrPlotSeries]:
    """Collect derivative plot series requested by `term.attrs.plot`.

    Expected DSL plot spec type:
      - `type = "traj_derivative"` (aliases supported)
      - required: `derivative_order` or `order`
      - optional: `name`
    """

    runtime = getattr(compiled, "runtime", None)
    if not isinstance(runtime, NLSRuntime):
        raise TypeError(
            "collect_trajectory_derivative_plot_series_from_term_attrs: "
            "compiled.runtime must be an NLSRuntime."
        )

    derivative_orders, name_overrides = _collect_derivative_plot_requests_from_term_attrs(
        runtime,
        term_indices=term_indices,
        strict=strict,
    )
    if len(derivative_orders) == 0:
        return []

    names = None if len(name_overrides) == 0 else name_overrides
    return collect_trajectory_derivative_plot_series(
        compiled,
        derivative_orders=tuple(derivative_orders),
        names=names,
        p=p,
        strict=strict,
    )


def collect_plot_series_from_compiled_term_attrs(
    compiled: Any,
    *,
    term_indices: Iterable[int] | None = None,
    p: Array | None = None,
    strict: bool = True,
) -> list[TermAttrPlotSeries]:
    """Collect all DSL-declared plot series from a compiled trajectory problem.

    This combines:
      - `state_traj` specs via runtime state collection
      - `traj_derivative` specs via trajectory derivative maps
    """

    runtime = getattr(compiled, "runtime", None)
    if not isinstance(runtime, NLSRuntime):
        raise TypeError(
            "collect_plot_series_from_compiled_term_attrs: "
            "compiled.runtime must be an NLSRuntime."
        )

    out = list(
        collect_plot_series_from_term_attrs(
            runtime,
            term_indices=term_indices,
            strict=strict,
        )
    )
    out.extend(
        collect_trajectory_derivative_plot_series_from_term_attrs(
            compiled,
            term_indices=term_indices,
            p=p,
            strict=strict,
        )
    )
    return out


def write_plot_series_csv(
    series: Iterable[TermAttrPlotSeries],
    path: str | Path,
) -> Path:
    """Write `TermAttrPlotSeries` to a wide-format CSV file.

    Output columns:
      - `x_axis`, `k`, `x`
      - one column per plotted line label (e.g. `joint_q[0]`, `joint_qdot[1]`)

    Each row corresponds to one `(x_axis, k, x)` sample.
    """

    series_list = list(series)
    if len(series_list) == 0:
        raise ValueError("write_plot_series_csv: no series were provided.")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _format_float(v: float) -> str:
        val = float(v)
        if np.isfinite(val):
            return f"{val:.12g}"
        return str(val)

    def _unique_label(raw: str, used: set[str]) -> str:
        base = str(raw).strip()
        if base == "":
            base = "value"
        if base not in used:
            used.add(base)
            return base
        i = 2
        while True:
            cand = f"{base}_{i}"
            if cand not in used:
                used.add(cand)
                return cand
            i += 1

    line_columns: list[str] = []
    line_col_for: dict[tuple[int, int], str] = {}
    used_labels: set[str] = set()
    for s_idx, item in enumerate(series_list):
        y = np.asarray(item.y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.ndim != 2:
            raise ValueError(
                "write_plot_series_csv: expected each series.y to be 2D. "
                f"Got shape {y.shape} for series {item.name!r}."
            )
        for j in range(int(y.shape[1])):
            col = _unique_label(item.line_label(j), used_labels)
            line_columns.append(col)
            line_col_for[(int(s_idx), int(j))] = col

    rows: dict[tuple[str, int, float], dict[str, float | int | str]] = {}
    for s_idx, item in enumerate(series_list):
        x = np.asarray(item.x, dtype=float).reshape(-1)
        y = np.asarray(item.y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.ndim != 2:
            raise ValueError(
                "write_plot_series_csv: expected each series.y to be 2D. "
                f"Got shape {y.shape} for series {item.name!r}."
            )

        n_rows = int(y.shape[0])
        n_cols = int(y.shape[1])
        if int(x.size) != n_rows:
            raise ValueError(
                "write_plot_series_csv: x/y length mismatch. "
                f"series={item.name!r}, x={x.size}, y_rows={n_rows}."
            )
        ks = tuple(int(k) for k in item.ks)
        if len(ks) != n_rows:
            raise ValueError(
                "write_plot_series_csv: ks/y length mismatch. "
                f"series={item.name!r}, ks={len(ks)}, y_rows={n_rows}."
            )

        axis = str(item.x_axis)
        for i in range(n_rows):
            key = (axis, int(ks[i]), float(x[i]))
            row = rows.get(key, None)
            if row is None:
                row = {"x_axis": axis, "k": int(ks[i]), "x": float(x[i])}
                rows[key] = row
            for j in range(n_cols):
                col = line_col_for[(int(s_idx), int(j))]
                y_val = float(y[i, j])
                if col in row:
                    prev = float(row[col])  # type: ignore[arg-type]
                    if not np.isclose(prev, y_val, rtol=1e-12, atol=1e-12, equal_nan=True):
                        raise ValueError(
                            "write_plot_series_csv: conflicting values detected while merging rows. "
                            f"key={key}, column={col!r}, prev={prev}, new={y_val}."
                        )
                else:
                    row[col] = y_val

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        columns = ["x_axis", "k", "x", *line_columns]
        writer.writerow(columns)

        for key in sorted(rows.keys(), key=lambda t: (str(t[0]), int(t[1]), float(t[2]))):
            row = rows[key]
            out_row: list[str] = []
            for col in columns:
                if col not in row:
                    out_row.append("")
                    continue
                val = row[col]
                if isinstance(val, float):
                    out_row.append(_format_float(val))
                else:
                    out_row.append(str(val))
            writer.writerow(out_row)

    return out_path


def _group_series_by_name(
    series: Sequence[TermAttrPlotSeries],
) -> tuple[list[str], dict[str, list[TermAttrPlotSeries]]]:
    group_order: list[str] = []
    groups: dict[str, list[TermAttrPlotSeries]] = {}
    for item in series:
        key = str(item.name)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(item)
    return group_order, groups


def _prioritize_group_order(
    group_order: Sequence[str],
    *,
    groups: Mapping[str, Sequence[TermAttrPlotSeries]],
    group_priorities: Sequence[str] | None,
) -> list[str]:
    if group_priorities is None:
        return list(group_order)

    out: list[str] = []
    seen: set[str] = set()
    for raw in group_priorities:
        key = str(raw)
        if key in groups and key not in seen:
            out.append(key)
            seen.add(key)

    for key in group_order:
        if key in seen:
            continue
        out.append(str(key))

    return out


def plot_series(
    series: Iterable[TermAttrPlotSeries],
    *,
    ax: Any = None,
    title: str | None = None,
    legend: bool = True,
    grid: bool = True,
    subplot_by: str | None = None,
    sharex: bool = True,
    group_priorities: Sequence[str] | None = None,
) -> tuple[Any, Any, list[TermAttrPlotSeries]]:
    """Plot pre-collected `TermAttrPlotSeries`.

    Returns `(fig, ax, series_list)`.
    """

    series_list = list(series)
    if len(series_list) == 0:
        raise ValueError("plot_series: no series were provided.")

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "plot_series requires matplotlib. Install project dependencies and retry."
        ) from e

    subplot_key = "" if subplot_by is None else str(subplot_by).strip().lower()
    if subplot_key not in ("", "name"):
        raise ValueError(
            "plot_series: subplot_by must be None or 'name'. "
            f"Got {subplot_by!r}."
        )

    if subplot_key != "name" and group_priorities is not None:
        raise ValueError(
            "plot_series: group_priorities can only be used when subplot_by='name'."
        )

    if subplot_key == "name":
        if ax is not None:
            raise ValueError("plot_series: ax cannot be used when subplot_by='name'.")

        group_order, groups = _group_series_by_name(series_list)
        group_order = _prioritize_group_order(
            group_order,
            groups=groups,
            group_priorities=group_priorities,
        )

        nrows = int(len(group_order))
        fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=bool(sharex))
        if nrows == 1:
            axes_list = [axes]
            axes_out: Any = axes
        else:
            axes_list = list(np.asarray(axes, dtype=object).reshape(-1))
            axes_out = axes

        has_time_axis_any = False
        for group_name, ax_i in zip(group_order, axes_list):
            items = groups[group_name]
            has_time_axis = any(item.x_axis == "time" for item in items)
            has_time_axis_any = has_time_axis_any or has_time_axis
            for item in items:
                for j in range(int(item.y.shape[1])):
                    ax_i.plot(item.x, item.y[:, j], label=item.line_label(j))

            ax_i.set_ylabel("value")
            ax_i.set_title(group_name)
            if grid:
                ax_i.grid(True, alpha=0.3)
            if legend:
                handles, _labels = ax_i.get_legend_handles_labels()
                if len(handles) > 0:
                    ax_i.legend()
            if not bool(sharex):
                ax_i.set_xlabel("time [s]" if has_time_axis else "k")

        if bool(sharex):
            axes_list[-1].set_xlabel("time [s]" if has_time_axis_any else "k")

        if title is None:
            title = "plot series"
        if str(title).strip() != "":
            fig.suptitle(str(title))

        return fig, axes_out, series_list

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = getattr(ax, "figure", None)
        if fig is None:
            raise TypeError("plot_series: ax must be a matplotlib Axes instance.")

    has_time_axis = False
    for item in series_list:
        if item.x_axis == "time":
            has_time_axis = True
        for i in range(int(item.y.shape[1])):
            ax.plot(item.x, item.y[:, i], label=item.line_label(i))

    ax.set_xlabel("time [s]" if has_time_axis else "k")
    ax.set_ylabel("value")
    if title is None:
        title = "plot series"
    if str(title).strip() != "":
        ax.set_title(str(title))
    if grid:
        ax.grid(True, alpha=0.3)
    if legend:
        handles, _labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend()

    return fig, ax, series_list


def plot_term_attrs(
    runtime: NLSRuntime,
    *,
    term_indices: Iterable[int] | None = None,
    strict: bool = True,
    ax: Any = None,
    title: str | None = None,
    legend: bool = True,
    grid: bool = True,
    subplot_by: str | None = None,
    sharex: bool = True,
    group_priorities: Sequence[str] | None = None,
) -> tuple[Any, Any, list[TermAttrPlotSeries]]:
    """Plot trajectories declared by `term.attrs.plot`.

    Returns `(fig, ax, series)`.
    """

    series = collect_plot_series_from_term_attrs(
        runtime,
        term_indices=term_indices,
        strict=strict,
    )
    if len(series) == 0:
        raise ValueError("plot_term_attrs: no plot series found in term.attrs.plot.")

    title_text = "term.attrs.plot" if title is None else title
    fig, axes, series_list = plot_series(
        series,
        ax=ax,
        title=title_text,
        legend=legend,
        grid=grid,
        subplot_by=subplot_by,
        sharex=sharex,
        group_priorities=group_priorities,
    )
    return fig, axes, series_list


__all__ = [
    "TermAttrPlotSeries",
    "collect_plot_series_from_term_attrs",
    "collect_plot_series_from_compiled_term_attrs",
    "collect_trajectory_derivative_plot_series",
    "collect_trajectory_derivative_plot_series_from_term_attrs",
    "plot_series",
    "plot_term_attrs",
    "write_plot_series_csv",
]
