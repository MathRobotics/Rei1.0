from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.expr.types import RuntimeContext, VariablePack
from ..core.state_cache import OwnerKey, StateKey
from ..core.state_schema import canonical_dtype_name, canonical_field_name
from ..problem import NLSProblem

Array = np.ndarray


def _canonical_constraint_kind(kind: Any) -> str:
    value = str(kind).strip().lower()
    if value in ("eq", "equality"):
        return "eq"
    if value in ("ineq", "inequality"):
        return "ineq"
    raise ValueError(
        f"constraint kind must be 'eq' or 'ineq'. Got {kind!r}."
    )


def _dedupe_required(keys: Iterable[StateKey]) -> list[StateKey]:
    out: list[StateKey] = []
    seen: set[StateKey] = set()
    for k in keys:
        if k in seen:
            continue
        out.append(k)
        seen.add(k)
    return out


def collect_required(problem: NLSProblem) -> list[StateKey]:
    req: list[StateKey] = []
    for expr, _cost in problem.terms:
        deps = getattr(expr, "deps", None)
        if callable(deps):
            req.extend(list(deps()))
    return _dedupe_required(req)


@dataclass(frozen=True)
class LinearizedTerm:
    term_index: int
    name: str
    attrs: dict[str, Any]
    residual: Array
    jacobian: Array


@dataclass(frozen=True)
class StackedTermSlice:
    term_index: int
    name: str
    attrs: dict[str, Any]
    row_start: int
    row_stop: int


@dataclass
class NLSRuntime:
    """Runtime holder for a compiled NLS problem.

    This keeps mutable execution concerns (pack/state/time/required) out of `NLSProblem`.
    """

    problem: NLSProblem
    ctx: RuntimeContext
    required: list[StateKey] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.required = _dedupe_required(self.required)

    @classmethod
    def from_problem(
        cls,
        problem: NLSProblem,
        *,
        state: Any = None,
        time: Any = None,
        required: Iterable[StateKey] | None = None,
    ) -> "NLSRuntime":
        ctx = RuntimeContext(
            pack=problem.variables,
            state=state,
            time=time,
            revision=int(getattr(time, "revision", 0)),
        )
        req = [] if required is None else _dedupe_required(required)
        return cls(problem=problem, ctx=ctx, required=req)

    @property
    def pack(self) -> VariablePack:
        return self.ctx.pack

    @property
    def state(self) -> Any:
        return self.ctx.state

    @property
    def time(self) -> Any:
        return self.ctx.time

    def required_list(self, required: Iterable[StateKey] | None = None) -> list[StateKey]:
        if required is None:
            if not self.required:
                self.required = collect_required(self.problem)
            return list(self.required)
        return _dedupe_required(required)

    def update_state_if_needed(self, *, required: Iterable[StateKey] | None = None) -> None:
        req = self.required_list(required)
        update_if_needed = getattr(self.state, "update_if_needed", None) if self.state is not None else None
        if callable(update_if_needed):
            update_if_needed(self.pack, time=self.time, required=req)

    def linearize(self, *, required: Iterable[StateKey] | None = None) -> tuple[Array, Array]:
        req = self.required_list(required)
        self.update_state_if_needed(required=req)
        return self.problem.linearize(ctx=self.ctx, time=self.time, required=req)

    def _normalize_term_indices(self, term_indices: Iterable[int] | None = None) -> list[int]:
        if term_indices is None:
            return list(range(len(self.problem.terms)))

        out: list[int] = []
        seen: set[int] = set()
        n_terms = len(self.problem.terms)
        for idx_raw in term_indices:
            idx = int(idx_raw)
            if idx < 0 or idx >= n_terms:
                raise IndexError(
                    "linearize_terms: term index out of range. "
                    f"Got {idx}, expected 0..{n_terms - 1}."
                )
            if idx in seen:
                continue
            out.append(idx)
            seen.add(idx)
        return out

    def _assemble_global_jacobian(self, *, term_name: str, expr_vars: Any, residual: Array, blocks: Any) -> Array:
        r = np.asarray(residual, dtype=float).reshape(-1)
        m = int(r.size)
        blocks_list = [np.asarray(B, dtype=float) for B in blocks]
        n_total = int(self.pack.n_total)
        slices = self.pack.slices

        if len(blocks_list) != len(expr_vars):
            raise ValueError(
                "linearize_terms: len(blocks) mismatch in term "
                f"{term_name!r}. blocks={len(blocks_list)}, vars={len(expr_vars)}."
            )

        Jg = np.zeros((m, n_total), dtype=float)
        for v, B in zip(expr_vars, blocks_list):
            if B.ndim != 2 or B.shape[0] != m:
                raise ValueError(
                    "linearize_terms: row mismatch in term "
                    f"{term_name!r}, var={getattr(v, 'name', '<unknown>')!r}. "
                    f"Expected ({m}, n), got {B.shape}."
                )
            var_name = getattr(v, "name", None)
            if not isinstance(var_name, str) or var_name == "":
                raise ValueError(
                    f"linearize_terms: invalid variable name in term {term_name!r}: {var_name!r}."
                )
            if var_name not in slices:
                raise ValueError(
                    f"linearize_terms: var {var_name!r} not found in VariablePack "
                    f"(term {term_name!r})."
                )
            s, e = slices[var_name]
            nv = int(e - s)
            if B.shape[1] != nv:
                raise ValueError(
                    "linearize_terms: col mismatch in term "
                    f"{term_name!r}, var={var_name!r}. Expected ({m}, {nv}), got {B.shape}."
                )
            Jg[:, s:e] = B
        return Jg

    def _linearize_term_arrays(
        self,
        *,
        required: Iterable[StateKey] | None = None,
        weighted: bool = True,
        term_indices: Iterable[int] | None = None,
    ) -> tuple[list[int], list[str], list[dict[str, Any]], list[Array], list[Array]]:
        req = self.required_list(required)
        self.update_state_if_needed(required=req)
        idxs = self._normalize_term_indices(term_indices)

        names: list[str] = []
        attrs_list: list[dict[str, Any]] = []
        residuals: list[Array] = []
        jacobians: list[Array] = []

        for idx in idxs:
            expr, cost = self.problem.terms[idx]
            term_name = self._term_display_name(idx)
            attrs = self.problem.term_attrs_at(idx)

            r_raw, blocks_raw = expr.eval(self.ctx)
            r_raw = np.asarray(r_raw, dtype=float).reshape(-1)
            blocks_raw = [np.asarray(B, dtype=float) for B in blocks_raw]

            if weighted:
                apply_cost = getattr(cost, "apply", None)
                if not callable(apply_cost):
                    raise TypeError(
                        f"linearize_terms: Cost object for term[{idx}] has no callable apply(r, blocks)."
                    )
                r_use, blocks_use = apply_cost(r_raw, blocks_raw)
                r_use = np.asarray(r_use, dtype=float).reshape(-1)
                J_use = self._assemble_global_jacobian(
                    term_name=term_name,
                    expr_vars=expr.vars,
                    residual=r_use,
                    blocks=blocks_use,
                )
            else:
                r_use = r_raw
                J_use = self._assemble_global_jacobian(
                    term_name=term_name,
                    expr_vars=expr.vars,
                    residual=r_use,
                    blocks=blocks_raw,
                )

            names.append(term_name)
            attrs_list.append(attrs)
            residuals.append(r_use)
            jacobians.append(J_use)
        return idxs, names, attrs_list, residuals, jacobians

    def _stack_term_arrays(
        self,
        *,
        idxs: list[int],
        names: list[str],
        attrs_list: list[dict[str, Any]],
        residuals: list[Array],
        jacobians: list[Array],
    ) -> tuple[Array, Array, list[StackedTermSlice]]:
        n_total = int(self.pack.n_total)
        if len(residuals) == 0:
            return np.zeros((0,), dtype=float), np.zeros((0, n_total), dtype=float), []

        rows = int(sum(np.asarray(r, dtype=float).size for r in residuals))
        r_all = np.empty((rows,), dtype=float)
        J_all = np.empty((rows, n_total), dtype=float)
        layout: list[StackedTermSlice] = []

        offset = 0
        for idx, name, attrs, r, J in zip(idxs, names, attrs_list, residuals, jacobians):
            r_vec = np.asarray(r, dtype=float).reshape(-1)
            J_mat = np.asarray(J, dtype=float)
            m = int(r_vec.size)
            if J_mat.ndim != 2 or J_mat.shape[0] != m or J_mat.shape[1] != n_total:
                raise ValueError(
                    "linearize_stacked_terms: internal shape mismatch for "
                    f"term[{idx}]. residual size={m}, jacobian shape={J_mat.shape}, "
                    f"expected (*, {n_total})."
                )
            row_start = int(offset)
            row_stop = int(offset + m)
            r_all[row_start:row_stop] = r_vec
            J_all[row_start:row_stop, :] = J_mat
            layout.append(
                StackedTermSlice(
                    term_index=int(idx),
                    name=str(name),
                    attrs=dict(attrs),
                    row_start=row_start,
                    row_stop=row_stop,
                )
            )
            offset = row_stop
        return r_all, J_all, layout

    def linearize_stacked_terms(
        self,
        *,
        required: Iterable[StateKey] | None = None,
        weighted: bool = True,
        term_indices: Iterable[int] | None = None,
    ) -> tuple[Array, Array]:
        """Linearize selected terms and return stacked residual/Jacobian without term objects."""

        idxs, names, attrs_list, residuals, jacobians = self._linearize_term_arrays(
            required=required,
            weighted=weighted,
            term_indices=term_indices,
        )
        r_all, J_all, _layout = self._stack_term_arrays(
            idxs=idxs,
            names=names,
            attrs_list=attrs_list,
            residuals=residuals,
            jacobians=jacobians,
        )
        return r_all, J_all

    def linearize_stacked_terms_with_layout(
        self,
        *,
        required: Iterable[StateKey] | None = None,
        weighted: bool = True,
        term_indices: Iterable[int] | None = None,
    ) -> tuple[Array, Array, list[StackedTermSlice]]:
        """Linearize selected terms and return stacked arrays with per-term row slices."""

        idxs, names, attrs_list, residuals, jacobians = self._linearize_term_arrays(
            required=required,
            weighted=weighted,
            term_indices=term_indices,
        )
        return self._stack_term_arrays(
            idxs=idxs,
            names=names,
            attrs_list=attrs_list,
            residuals=residuals,
            jacobians=jacobians,
        )

    def linearize_terms(
        self,
        *,
        required: Iterable[StateKey] | None = None,
        weighted: bool = True,
        term_indices: Iterable[int] | None = None,
    ) -> list[LinearizedTerm]:
        """Linearize each selected term separately.

        Returns a list of `LinearizedTerm` entries preserving requested term order.
        """

        idxs, names, attrs_list, residuals, jacobians = self._linearize_term_arrays(
            required=required,
            weighted=weighted,
            term_indices=term_indices,
        )
        out: list[LinearizedTerm] = []
        for idx, term_name, attrs, r_use, J_use in zip(idxs, names, attrs_list, residuals, jacobians):
            out.append(
                LinearizedTerm(
                    term_index=idx,
                    name=term_name,
                    attrs=attrs,
                    residual=r_use.copy(),
                    jacobian=J_use.copy(),
                )
            )
        return out

    def linearize_constraint_terms(
        self,
        *,
        required: Iterable[StateKey] | None = None,
        kind: str | None = None,
        weighted: bool = False,
    ) -> list[LinearizedTerm]:
        idxs = self.find_constraint_term_indices(kind=kind)
        return self.linearize_terms(required=required, weighted=weighted, term_indices=idxs)

    def collect_state_traj(
        self,
        *,
        owner_type: str,
        owner_name: str,
        dtype: str,
        field: str,
        ks: Iterable[int] | None = None,
        k0: int = 0,
        k1: int | None = None,
        frame: str | None = None,
        rel_frame: str | None = None,
        expected_dim: int | None = None,
    ) -> Array:
        """Collect a stacked state trajectory for one `StateKey` family.

        Returns an array with shape `(len(ks), dim)`.
        """

        owner_type = str(owner_type).strip()
        owner_name = str(owner_name).strip()
        dtype = canonical_dtype_name(str(dtype).strip())
        field = canonical_field_name(str(field).strip())
        if owner_type == "" or owner_name == "" or dtype == "" or field == "":
            raise ValueError("collect_state_traj: owner_type/owner_name/dtype/field must be non-empty.")

        if ks is not None and (int(k0) != 0 or k1 is not None):
            raise ValueError("collect_state_traj: use either `ks` or (`k0`, `k1`), not both.")

        if expected_dim is not None:
            expected_dim = int(expected_dim)
            if expected_dim < 0:
                raise ValueError(f"collect_state_traj: expected_dim must be >= 0, got {expected_dim}.")

        if ks is None:
            start = int(k0)
            if start < 0:
                raise ValueError(f"collect_state_traj: k0 must be >= 0, got {start}.")
            if k1 is None:
                if self.time is not None and hasattr(self.time, "N"):
                    stop = int(self.time.N) + 1
                else:
                    stop = start + 1
            else:
                stop = int(k1)
            if stop < start:
                raise ValueError(f"collect_state_traj: expected k1 >= k0, got k0={start}, k1={stop}.")
            ks_list = list(range(start, stop))
        else:
            ks_list = [int(k) for k in ks]
            if any(k < 0 for k in ks_list):
                raise ValueError("collect_state_traj: all ks must be >= 0.")

        owner = OwnerKey(owner_type=owner_type, owner_name=owner_name)
        required = [
            StateKey(
                k=int(k),
                owner=owner,
                dtype=dtype,
                field=field,
                frame=frame,
                rel_frame=rel_frame,
            )
            for k in ks_list
        ]

        if len(required) == 0:
            cols = 0 if expected_dim is None else expected_dim
            return np.zeros((0, cols), dtype=float)

        self.update_state_if_needed(required=required)
        rows: list[Array] = []
        for key in required:
            if self.state is None:
                raise ValueError("collect_state_traj: runtime.state is None.")
            v = np.asarray(self.state.get(key), dtype=float).reshape(-1)
            rows.append(v)

        dim = int(rows[0].size) if expected_dim is None else int(expected_dim)
        for i, v in enumerate(rows):
            if v.size != dim:
                raise ValueError(
                    "collect_state_traj: state vector size mismatch. "
                    f"k={ks_list[i]}, expected {dim}, got {v.size}."
                )
        if len(rows) == 0:
            return np.zeros((0, dim), dtype=float)
        return np.vstack(rows)

    def cost_value(self, *, required: Iterable[StateKey] | None = None) -> float:
        r_all, _ = self.linearize(required=required)
        return float(r_all @ r_all)

    def _term_display_name(self, index: int) -> str:
        expr, _cost = self.problem.terms[index]
        name = getattr(expr, "name", None)
        if isinstance(name, str) and name:
            return name
        return expr.__class__.__name__

    def term_attrs(self, term: int | str) -> dict[str, Any]:
        idx = self._resolve_term_index(term)
        return self.problem.term_attrs_at(idx)

    def find_term_indices(self, *, attr: str, value: Any = True) -> list[int]:
        return self.problem.find_terms_by_attr(attr, value=value)

    def find_constraint_term_indices(self, *, kind: str | None = None) -> list[int]:
        kind_norm = None if kind is None else _canonical_constraint_kind(kind)
        idxs: list[int] = []

        for i, attrs in enumerate(self.problem.term_attrs):
            is_constraint = bool(attrs.get("is_constraint", False))
            term_kind_raw = attrs.get("constraint_kind", None)
            term_kind = None if term_kind_raw is None else _canonical_constraint_kind(term_kind_raw)

            if not is_constraint and term_kind is None:
                continue
            if kind_norm is not None and term_kind != kind_norm:
                continue
            idxs.append(i)

        return idxs

    def _resolve_term_index(self, term: int | str) -> int:
        if isinstance(term, int):
            idx = int(term)
            if idx < 0 or idx >= len(self.problem.terms):
                raise IndexError(
                    f"set_cost_weight: term index out of range: {idx}. "
                    f"Expected 0..{len(self.problem.terms) - 1}."
                )
            return idx

        term_name = str(term)
        if term_name == "":
            raise ValueError("set_cost_weight: term name must be non-empty.")

        matches = [
            i
            for i, (expr, _cost) in enumerate(self.problem.terms)
            if getattr(expr, "name", None) == term_name
        ]
        if len(matches) == 0:
            raise ValueError(f"set_cost_weight: no term found with name={term_name!r}.")
        if len(matches) > 1:
            idxs = ", ".join(str(i) for i in matches)
            raise ValueError(
                f"set_cost_weight: multiple terms matched name={term_name!r} at indices [{idxs}]. "
                "Use an explicit term index."
            )
        return matches[0]

    def set_cost_weight(self, term: int | str, w: Any) -> int:
        """Update one term's cost weight and invalidate linearization cache.

        `term` can be either the term index or the term expression name.
        The target cost must implement `set_weight(w)`.
        """

        idx = self._resolve_term_index(term)
        _expr, cost = self.problem.terms[idx]
        set_weight = getattr(cost, "set_weight", None)
        if not callable(set_weight):
            raise TypeError(
                f"set_cost_weight: term[{idx}] '{self._term_display_name(idx)}' "
                f"uses cost type '{type(cost).__name__}' which does not support runtime weight updates."
            )

        set_weight(w)
        self.problem.invalidate_cache()
        return idx

    def set_cost_weight_by_attr(
        self,
        *,
        attr: str,
        value: Any = True,
        w: Any,
        require_match: bool = True,
    ) -> list[int]:
        """Update all term weights matching one term attribute.

        Returns the updated term indices.
        """

        idxs = self.find_term_indices(attr=attr, value=value)
        if len(idxs) == 0:
            if require_match:
                raise ValueError(
                    f"set_cost_weight_by_attr: no term matched attr={attr!r}, value={value!r}."
                )
            return []

        set_weight_fns: list[tuple[int, Any]] = []
        for idx in idxs:
            _expr, cost = self.problem.terms[idx]
            set_weight = getattr(cost, "set_weight", None)
            if not callable(set_weight):
                raise TypeError(
                    f"set_cost_weight_by_attr: term[{idx}] '{self._term_display_name(idx)}' "
                    f"uses cost type '{type(cost).__name__}' which does not support runtime weight updates."
                )
            set_weight_fns.append((idx, set_weight))

        for _idx, set_weight in set_weight_fns:
            set_weight(w)

        self.problem.invalidate_cache()
        return idxs

    def set_cost_weight_by_constraint(
        self,
        *,
        kind: str | None = None,
        w: Any,
        require_match: bool = True,
    ) -> list[int]:
        idxs = self.find_constraint_term_indices(kind=kind)
        if len(idxs) == 0:
            if require_match:
                raise ValueError(
                    f"set_cost_weight_by_constraint: no term matched kind={kind!r}."
                )
            return []

        set_weight_fns: list[tuple[int, Any]] = []
        for idx in idxs:
            _expr, cost = self.problem.terms[idx]
            set_weight = getattr(cost, "set_weight", None)
            if not callable(set_weight):
                raise TypeError(
                    f"set_cost_weight_by_constraint: term[{idx}] '{self._term_display_name(idx)}' "
                    f"uses cost type '{type(cost).__name__}' which does not support runtime weight updates."
                )
            set_weight_fns.append((idx, set_weight))

        for _idx, set_weight in set_weight_fns:
            set_weight(w)

        self.problem.invalidate_cache()
        return idxs

__all__ = [
    "NLSRuntime",
    "LinearizedTerm",
    "StackedTermSlice",
    "collect_required",
]
