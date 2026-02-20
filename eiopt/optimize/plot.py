from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.state_cache import OwnerKey, StateKey
from ..core.state_schema import canonical_dtype_name, canonical_field_name
from .runtime import NLSRuntime

Array = np.ndarray


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

    Supported spec type:
      - `type = "state_traj"`
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
            typ = str(spec.get("type", spec.get("kind", "state_traj"))).strip().lower()
            if typ != "state_traj":
                if strict:
                    raise ValueError(
                        f"{where}: unsupported plot type {typ!r}. "
                        "Supported type is exactly 'state_traj'."
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


def plot_term_attrs(
    runtime: NLSRuntime,
    *,
    term_indices: Iterable[int] | None = None,
    strict: bool = True,
    ax: Any = None,
    title: str | None = None,
    legend: bool = True,
    grid: bool = True,
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

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "plot_term_attrs requires matplotlib. Install project dependencies and retry."
        ) from e

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = getattr(ax, "figure", None)
        if fig is None:
            raise TypeError("plot_term_attrs: ax must be a matplotlib Axes instance.")

    has_time_axis = False
    for item in series:
        if item.x_axis == "time":
            has_time_axis = True
        for i in range(int(item.y.shape[1])):
            ax.plot(item.x, item.y[:, i], label=item.line_label(i))

    ax.set_xlabel("time [s]" if has_time_axis else "k")
    ax.set_ylabel("value")
    if title is None:
        title = "term.attrs.plot"
    if str(title).strip() != "":
        ax.set_title(str(title))
    if grid:
        ax.grid(True, alpha=0.3)
    if legend:
        handles, _labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend()

    return fig, ax, series


__all__ = [
    "TermAttrPlotSeries",
    "collect_plot_series_from_term_attrs",
    "plot_term_attrs",
]
