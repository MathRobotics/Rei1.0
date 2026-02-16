from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..core.state_cache import StateKey
from .matrix_scaling import scale_matrix_with_projection_svd
from .runtime import LinearizedTerm, ProblemRuntime, StackedTermSlice
from .term import Variable, VariablePack

Array = np.ndarray


def _set_pack_x(pack: VariablePack, x: Array) -> None:
    x_target = np.asarray(x, dtype=float).reshape(-1)
    x_cur = np.asarray(pack.get(), dtype=float).reshape(-1)
    if x_target.shape != x_cur.shape:
        raise ValueError(f"_set_pack_x: shape mismatch {x_target.shape} vs {x_cur.shape}.")
    if np.array_equal(x_target, x_cur):
        return
    pack.apply_dx(x_target - x_cur)


def _normalize_term_indices(*, n_terms: int, term_indices: Iterable[int] | None) -> tuple[int, ...]:
    if term_indices is None:
        return tuple(range(int(n_terms)))
    out: list[int] = []
    seen: set[int] = set()
    n = int(n_terms)
    for idx_raw in term_indices:
        idx = int(idx_raw)
        if idx < 0 or idx >= n:
            raise IndexError(
                f"_normalize_term_indices: index out of range: {idx}. Expected 0..{n - 1}."
            )
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return tuple(out)


def _max_abs(x: Array) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


@dataclass
class NullspaceReducedRuntime:
    full_runtime: ProblemRuntime
    objective_term_indices: tuple[int, ...]
    x_particular: Array
    nullspace_basis: Array
    z_var_name: str = "z_nullspace"

    def __post_init__(self) -> None:
        n_full = int(self.full_runtime.pack.n_total)
        self.x_particular = np.asarray(self.x_particular, dtype=float).reshape(-1)
        if self.x_particular.size != n_full:
            raise ValueError(
                "NullspaceReducedRuntime: x_particular size mismatch. "
                f"Expected {n_full}, got {self.x_particular.size}."
            )

        self.nullspace_basis = np.asarray(self.nullspace_basis, dtype=float)
        if self.nullspace_basis.ndim != 2:
            raise ValueError(
                "NullspaceReducedRuntime: nullspace_basis must be 2D, "
                f"got shape {self.nullspace_basis.shape}."
            )
        if self.nullspace_basis.shape[0] != n_full:
            raise ValueError(
                "NullspaceReducedRuntime: nullspace_basis row mismatch. "
                f"Expected {n_full}, got {self.nullspace_basis.shape[0]}."
            )

        n_reduced = int(self.nullspace_basis.shape[1])
        z_name = str(self.z_var_name).strip()
        if z_name == "":
            raise ValueError("NullspaceReducedRuntime: z_var_name must be non-empty.")
        z_var = Variable(name=z_name, x=np.zeros((n_reduced,), dtype=float))
        self._pack = VariablePack([z_var])

        self.objective_term_indices = _normalize_term_indices(
            n_terms=len(self.full_runtime.problem.terms),
            term_indices=self.objective_term_indices,
        )
        self._objective_term_index_set = set(self.objective_term_indices)

        x_cur = np.asarray(self.full_runtime.pack.get(), dtype=float).reshape(-1)
        z0 = self.project(x_cur)
        _set_pack_x(self._pack, z0)
        self._sync_full_from_reduced()

    @property
    def pack(self) -> VariablePack:
        return self._pack

    def required_list(self, required: Iterable[StateKey] | None = None) -> list[StateKey]:
        return self.full_runtime.required_list(required)

    def lift(self, z: Array) -> Array:
        z_vec = np.asarray(z, dtype=float).reshape(-1)
        n_reduced = int(self.nullspace_basis.shape[1])
        if z_vec.size != n_reduced:
            raise ValueError(
                "NullspaceReducedRuntime.lift: z size mismatch. "
                f"Expected {n_reduced}, got {z_vec.size}."
            )
        return np.asarray(self.x_particular + self.nullspace_basis @ z_vec, dtype=float).reshape(-1)

    def project(self, x: Array) -> Array:
        x_vec = np.asarray(x, dtype=float).reshape(-1)
        n_full = int(self.nullspace_basis.shape[0])
        if x_vec.size != n_full:
            raise ValueError(
                "NullspaceReducedRuntime.project: x size mismatch. "
                f"Expected {n_full}, got {x_vec.size}."
            )
        if self.nullspace_basis.shape[1] == 0:
            return np.zeros((0,), dtype=float)
        z, *_ = np.linalg.lstsq(self.nullspace_basis, x_vec - self.x_particular, rcond=None)
        return np.asarray(z, dtype=float).reshape(-1)

    def _sync_full_from_reduced(self) -> None:
        z = np.asarray(self._pack.get(), dtype=float).reshape(-1)
        x = self.lift(z)
        _set_pack_x(self.full_runtime.pack, x)

    def update_state_if_needed(self, *, required: Iterable[StateKey] | None = None) -> None:
        req = self.required_list(required)
        self._sync_full_from_reduced()
        self.full_runtime.update_state_if_needed(required=req)

    def _selected_full_term_indices(self, term_indices: Iterable[int] | None = None) -> tuple[int, ...]:
        if term_indices is None:
            return self.objective_term_indices

        full = _normalize_term_indices(
            n_terms=len(self.full_runtime.problem.terms),
            term_indices=term_indices,
        )
        invalid = [idx for idx in full if idx not in self._objective_term_index_set]
        if len(invalid) > 0:
            invalid_str = ", ".join(str(i) for i in invalid)
            raise ValueError(
                "NullspaceReducedRuntime.linearize_terms: term_indices must reference objective terms "
                f"in global problem indexing. Non-objective index(es): [{invalid_str}]."
            )
        return full

    def _linearize_full_terms(
        self,
        *,
        required: Iterable[StateKey] | None,
        weighted: bool,
        term_indices: Iterable[int] | None,
    ) -> list[LinearizedTerm]:
        req = self.required_list(required)
        self._sync_full_from_reduced()
        full_idxs = self._selected_full_term_indices(term_indices)
        return self.full_runtime.linearize_terms(
            required=req,
            weighted=weighted,
            term_indices=full_idxs,
        )

    def _linearize_full_stacked(
        self,
        *,
        required: Iterable[StateKey] | None,
        weighted: bool,
        term_indices: Iterable[int] | None,
    ) -> tuple[Array, Array]:
        req = self.required_list(required)
        self._sync_full_from_reduced()
        full_idxs = self._selected_full_term_indices(term_indices)
        return self.full_runtime.linearize_stacked_terms(
            required=req,
            weighted=weighted,
            term_indices=full_idxs,
        )

    def _linearize_full_stacked_with_layout(
        self,
        *,
        required: Iterable[StateKey] | None,
        weighted: bool,
        term_indices: Iterable[int] | None,
    ) -> tuple[Array, Array, list[StackedTermSlice]]:
        req = self.required_list(required)
        self._sync_full_from_reduced()
        full_idxs = self._selected_full_term_indices(term_indices)
        return self.full_runtime.linearize_stacked_terms_with_layout(
            required=req,
            weighted=weighted,
            term_indices=full_idxs,
        )

    def linearize_stacked_terms(
        self,
        *,
        required: Iterable[StateKey] | None = None,
        weighted: bool = True,
        term_indices: Iterable[int] | None = None,
    ) -> tuple[Array, Array]:
        r_full, J_full = self._linearize_full_stacked(
            required=required,
            weighted=weighted,
            term_indices=term_indices,
        )
        return np.asarray(r_full, dtype=float).reshape(-1), np.asarray(J_full, dtype=float) @ self.nullspace_basis

    def linearize_stacked_terms_with_layout(
        self,
        *,
        required: Iterable[StateKey] | None = None,
        weighted: bool = True,
        term_indices: Iterable[int] | None = None,
    ) -> tuple[Array, Array, list[StackedTermSlice]]:
        r_full, J_full, layout = self._linearize_full_stacked_with_layout(
            required=required,
            weighted=weighted,
            term_indices=term_indices,
        )
        J_reduced = np.asarray(J_full, dtype=float) @ self.nullspace_basis
        return np.asarray(r_full, dtype=float).reshape(-1), J_reduced, list(layout)

    def linearize_terms(
        self,
        *,
        required: Iterable[StateKey] | None = None,
        weighted: bool = True,
        term_indices: Iterable[int] | None = None,
    ) -> list[LinearizedTerm]:
        terms = self._linearize_full_terms(
            required=required,
            weighted=weighted,
            term_indices=term_indices,
        )
        out: list[LinearizedTerm] = []
        for term in terms:
            out.append(
                LinearizedTerm(
                    term_index=int(term.term_index),
                    name=str(term.name),
                    attrs=dict(term.attrs),
                    residual=np.asarray(term.residual, dtype=float).reshape(-1).copy(),
                    jacobian=np.asarray(term.jacobian, dtype=float) @ self.nullspace_basis,
                )
            )
        return out

    def linearize(self, *, required: Iterable[StateKey] | None = None) -> tuple[Array, Array]:
        # Fast path for solver loops: avoid term object creation on both runtimes.
        r_red, J_red = self.linearize_stacked_terms(
            required=required,
            weighted=True,
            term_indices=None,
        )
        return r_red, J_red

    def cost_value(self, *, required: Iterable[StateKey] | None = None) -> float:
        r, _ = self.linearize(required=required)
        return float(r @ r)


@dataclass(frozen=True)
class NullspaceEqualityReduction:
    runtime: NullspaceReducedRuntime
    full_runtime: ProblemRuntime
    eq_term_indices: tuple[int, ...]
    objective_term_indices: tuple[int, ...]
    constraint_jacobian: Array
    constraint_offset: Array
    x_particular: Array
    nullspace_basis: Array
    rank: int
    feasibility_residual_norm: float

    def lift(self, z: Array) -> Array:
        return self.runtime.lift(z)

    def project(self, x: Array) -> Array:
        return self.runtime.project(x)


def build_nullspace_equality_reduction(
    runtime: ProblemRuntime,
    *,
    eq_term_indices: Iterable[int] | None = None,
    objective_term_indices: Iterable[int] | None = None,
    required: Iterable[StateKey] | None = None,
    z_var_name: str = "z_nullspace",
    use_scaling: bool = True,
    svd_rtol: float = 1e-10,
    feasibility_tol: float = 1e-8,
    check_linearity: bool = True,
    linearity_samples: int = 2,
    linearity_step: float = 1e-6,
    linearity_atol: float = 1e-9,
    linearity_rtol: float = 1e-6,
    linearity_seed: int = 0,
) -> NullspaceEqualityReduction:
    if not isinstance(runtime, ProblemRuntime):
        raise TypeError(
            "build_nullspace_equality_reduction: runtime must be ProblemRuntime."
        )
    use_scaling_b = bool(use_scaling)
    svd_rtol_f = float(svd_rtol)
    if svd_rtol_f < 0.0:
        raise ValueError(
            f"build_nullspace_equality_reduction: svd_rtol must be >= 0, got {svd_rtol_f}."
        )
    feasibility_tol_f = float(feasibility_tol)
    if feasibility_tol_f < 0.0:
        raise ValueError(
            "build_nullspace_equality_reduction: feasibility_tol must be >= 0, "
            f"got {feasibility_tol_f}."
        )
    linearity_step_f = float(linearity_step)
    if linearity_step_f < 0.0:
        raise ValueError(
            f"build_nullspace_equality_reduction: linearity_step must be >= 0, got {linearity_step_f}."
        )
    linearity_atol_f = float(linearity_atol)
    if linearity_atol_f < 0.0:
        raise ValueError(
            f"build_nullspace_equality_reduction: linearity_atol must be >= 0, got {linearity_atol_f}."
        )
    linearity_rtol_f = float(linearity_rtol)
    if linearity_rtol_f < 0.0:
        raise ValueError(
            f"build_nullspace_equality_reduction: linearity_rtol must be >= 0, got {linearity_rtol_f}."
        )
    linearity_samples_i = int(linearity_samples)
    if linearity_samples_i < 0:
        raise ValueError(
            "build_nullspace_equality_reduction: linearity_samples must be >= 0, "
            f"got {linearity_samples_i}."
        )

    n_terms = len(runtime.problem.terms)
    eq_idxs_auto = _normalize_term_indices(
        n_terms=n_terms,
        term_indices=runtime.find_constraint_term_indices(kind="eq"),
    )
    if eq_term_indices is None:
        eq_idxs = eq_idxs_auto
    else:
        eq_idxs = _normalize_term_indices(n_terms=n_terms, term_indices=eq_term_indices)
        eq_auto_set = set(eq_idxs_auto)
        invalid = [idx for idx in eq_idxs if idx not in eq_auto_set]
        if len(invalid) > 0:
            invalid_details: list[str] = []
            for idx in invalid:
                attrs = runtime.problem.term_attrs_at(idx)
                kind = attrs.get("constraint_kind", None)
                is_constraint = bool(attrs.get("is_constraint", False))
                invalid_details.append(
                    f"{idx}(constraint_kind={kind!r}, is_constraint={is_constraint})"
                )
            detail = ", ".join(invalid_details)
            raise ValueError(
                "build_nullspace_equality_reduction: eq_term_indices must reference terms with "
                "constraint.kind='eq'. Invalid index(es): "
                f"[{detail}]."
            )

    if objective_term_indices is None:
        eq_set = set(eq_idxs)
        obj_idxs = tuple(i for i in range(n_terms) if i not in eq_set)
    else:
        obj_idxs = _normalize_term_indices(n_terms=n_terms, term_indices=objective_term_indices)

    req = runtime.required_list(required)
    n_full = int(runtime.pack.n_total)
    x_cur = np.asarray(runtime.pack.get(), dtype=float).reshape(-1)
    if x_cur.size != n_full:
        raise ValueError(
            "build_nullspace_equality_reduction: runtime pack size mismatch. "
            f"Expected {n_full}, got {x_cur.size}."
        )

    if len(eq_idxs) == 0:
        A_eq = np.zeros((0, n_full), dtype=float)
        c_eq = np.zeros((0,), dtype=float)
        x_particular = x_cur.copy()
        Z = np.eye(n_full, dtype=float)
        rank = 0
        feas_norm = 0.0
    else:
        eq_terms = runtime.linearize_terms(required=req, weighted=False, term_indices=eq_idxs)
        if len(eq_terms) == 0:
            raise RuntimeError(
                "build_nullspace_equality_reduction: no equality terms were linearized."
            )
        A_eq = np.vstack([np.asarray(t.jacobian, dtype=float) for t in eq_terms])
        r_eq = np.concatenate([np.asarray(t.residual, dtype=float).reshape(-1) for t in eq_terms], axis=0)
        if A_eq.shape[1] != n_full:
            raise ValueError(
                "build_nullspace_equality_reduction: equality Jacobian column mismatch. "
                f"Expected {n_full}, got {A_eq.shape[1]}."
            )
        c_eq = np.asarray(r_eq - A_eq @ x_cur, dtype=float).reshape(-1)
        x_particular, *_ = np.linalg.lstsq(A_eq, -c_eq, rcond=None)
        x_particular = np.asarray(x_particular, dtype=float).reshape(-1)

        feas_res = np.asarray(A_eq @ x_particular + c_eq, dtype=float).reshape(-1)
        feas_norm = float(np.linalg.norm(feas_res))
        if feas_norm > feasibility_tol_f:
            raise ValueError(
                "build_nullspace_equality_reduction: equality constraints are infeasible "
                f"under linearized model (||A x + c||={feas_norm:.3e}, tol={feasibility_tol_f:.3e})."
            )

        if use_scaling_b:
            row_basis, rank = scale_matrix_with_projection_svd(A_eq.T, svd_rtol=svd_rtol_f)
            if rank <= 0:
                Z = np.eye(n_full, dtype=float)
            elif rank < n_full:
                q, _ = np.linalg.qr(row_basis, mode="complete")
                Z = np.asarray(q[:, rank:], dtype=float)
            else:
                Z = np.zeros((n_full, 0), dtype=float)
        else:
            _u, svals, vt = np.linalg.svd(A_eq, full_matrices=True)
            if svals.size == 0:
                rank = 0
            else:
                threshold = svd_rtol_f * max(A_eq.shape) * float(svals[0])
                rank = int(np.sum(svals > threshold))
            if rank < n_full:
                Z = np.asarray(vt[rank:, :], dtype=float).T
            else:
                Z = np.zeros((n_full, 0), dtype=float)

        if bool(check_linearity) and linearity_samples_i > 0 and linearity_step_f > 0.0:
            x_restore = np.asarray(runtime.pack.get(), dtype=float).reshape(-1).copy()
            rng = np.random.default_rng(int(linearity_seed))
            scale = max(1.0, float(np.linalg.norm(x_restore)))
            step_norm = linearity_step_f * scale
            try:
                for sample_id in range(linearity_samples_i):
                    direction = np.asarray(rng.normal(size=n_full), dtype=float).reshape(-1)
                    norm_dir = float(np.linalg.norm(direction))
                    if norm_dir <= 0.0:
                        continue
                    direction = direction / norm_dir
                    x_trial = np.asarray(x_restore + step_norm * direction, dtype=float).reshape(-1)
                    _set_pack_x(runtime.pack, x_trial)
                    trial_terms = runtime.linearize_terms(
                        required=req,
                        weighted=False,
                        term_indices=eq_idxs,
                    )
                    A_trial = np.vstack([np.asarray(t.jacobian, dtype=float) for t in trial_terms])
                    r_trial = np.concatenate(
                        [np.asarray(t.residual, dtype=float).reshape(-1) for t in trial_terms],
                        axis=0,
                    )
                    r_affine = np.asarray(A_eq @ x_trial + c_eq, dtype=float).reshape(-1)

                    jac_err = _max_abs(A_trial - A_eq)
                    jac_scale = max(1.0, _max_abs(A_eq), _max_abs(A_trial))
                    jac_tol = linearity_atol_f + linearity_rtol_f * jac_scale
                    if jac_err > jac_tol:
                        raise ValueError(
                            "build_nullspace_equality_reduction: linearity check failed for equality Jacobian. "
                            f"sample={sample_id}, max|A_trial-A_ref|={jac_err:.3e}, tol={jac_tol:.3e}."
                        )

                    residual_err = _max_abs(r_trial - r_affine)
                    residual_scale = max(1.0, _max_abs(r_trial), _max_abs(r_affine))
                    residual_tol = linearity_atol_f + linearity_rtol_f * residual_scale
                    if residual_err > residual_tol:
                        raise ValueError(
                            "build_nullspace_equality_reduction: linearity check failed for equality residual affine model. "
                            f"sample={sample_id}, max|r_trial-(A_ref x + c_ref)|={residual_err:.3e}, "
                            f"tol={residual_tol:.3e}."
                        )
            finally:
                _set_pack_x(runtime.pack, x_restore)

    reduced_runtime = NullspaceReducedRuntime(
        full_runtime=runtime,
        objective_term_indices=tuple(obj_idxs),
        x_particular=x_particular.copy(),
        nullspace_basis=Z.copy(),
        z_var_name=z_var_name,
    )
    return NullspaceEqualityReduction(
        runtime=reduced_runtime,
        full_runtime=runtime,
        eq_term_indices=tuple(eq_idxs),
        objective_term_indices=tuple(obj_idxs),
        constraint_jacobian=A_eq.copy(),
        constraint_offset=c_eq.copy(),
        x_particular=x_particular.copy(),
        nullspace_basis=Z.copy(),
        rank=int(rank),
        feasibility_residual_norm=float(feas_norm),
    )
