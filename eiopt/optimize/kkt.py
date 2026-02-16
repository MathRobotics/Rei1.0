from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..core.state_cache import StateKey
from .runtime import NLSRuntime

Array = np.ndarray


@dataclass(frozen=True)
class KKTCheckResult:
    ok: bool
    message: str
    stationarity_inf: float
    eq_violation_inf: float
    ineq_violation_inf: float
    complementarity_inf: float
    dual_violation_inf: float
    n_vars: int
    n_objective_rows: int
    n_eq_rows: int
    n_ineq_rows: int
    n_active_ineq_rows: int
    lambda_eq: Array
    mu_ineq: Array


def _normalize_term_indices(
    *,
    n_terms: int,
    term_indices: Iterable[int] | None,
    name: str,
) -> tuple[int, ...]:
    if term_indices is None:
        return tuple()
    out: list[int] = []
    seen: set[int] = set()
    n = int(n_terms)
    for idx_raw in term_indices:
        idx = int(idx_raw)
        if idx < 0 or idx >= n:
            raise IndexError(f"{name}: index out of range: {idx}. Expected 0..{n - 1}.")
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return tuple(out)


def _inequality_sign_to_standardized(sense: str) -> tuple[str, float]:
    v = str(sense).strip().lower()
    if v in {"<=", "<", "le", "less_equal", "less-than-or-equal"}:
        return "<=", 1.0
    if v in {">=", ">", "ge", "greater_equal", "greater-than-or-equal"}:
        return ">=", -1.0
    raise ValueError(
        "check_kkt_conditions: ineq_sense must be one of '<=' or '>=' (aliases allowed). "
        f"Got {sense!r}."
    )


def check_kkt_residuals(
    *,
    grad_objective: Array,
    eq_residual: Array | None = None,
    eq_jacobian: Array | None = None,
    ineq_residual: Array | None = None,
    ineq_jacobian: Array | None = None,
    n_objective_rows: int = 0,
    active_tol: float = 1e-8,
    stationarity_tol: float = 1e-6,
    eq_tol: float = 1e-8,
    ineq_tol: float = 1e-8,
    complementarity_tol: float = 1e-8,
    dual_tol: float = 1e-10,
) -> KKTCheckResult:
    """Check first-order KKT residuals from generic linearized quantities.

    Conventions:
      - Equality constraints: `c_eq(x) = 0`
      - Inequality constraints: `c_in(x) <= 0`
    """

    active_tol_f = float(active_tol)
    stationarity_tol_f = float(stationarity_tol)
    eq_tol_f = float(eq_tol)
    ineq_tol_f = float(ineq_tol)
    complementarity_tol_f = float(complementarity_tol)
    dual_tol_f = float(dual_tol)
    if active_tol_f < 0.0:
        raise ValueError(f"check_kkt_residuals: active_tol must be >= 0, got {active_tol_f}.")
    if stationarity_tol_f < 0.0:
        raise ValueError(f"check_kkt_residuals: stationarity_tol must be >= 0, got {stationarity_tol_f}.")
    if eq_tol_f < 0.0:
        raise ValueError(f"check_kkt_residuals: eq_tol must be >= 0, got {eq_tol_f}.")
    if ineq_tol_f < 0.0:
        raise ValueError(f"check_kkt_residuals: ineq_tol must be >= 0, got {ineq_tol_f}.")
    if complementarity_tol_f < 0.0:
        raise ValueError(
            f"check_kkt_residuals: complementarity_tol must be >= 0, got {complementarity_tol_f}."
        )
    if dual_tol_f < 0.0:
        raise ValueError(f"check_kkt_residuals: dual_tol must be >= 0, got {dual_tol_f}.")

    n_obj_rows = int(n_objective_rows)
    if n_obj_rows < 0:
        raise ValueError(f"check_kkt_residuals: n_objective_rows must be >= 0, got {n_obj_rows}.")

    grad_obj = np.asarray(grad_objective, dtype=float).reshape(-1)
    n_vars = int(grad_obj.size)

    r_eq = np.zeros((0,), dtype=float) if eq_residual is None else np.asarray(eq_residual, dtype=float).reshape(-1)
    g_in = np.zeros((0,), dtype=float) if ineq_residual is None else np.asarray(ineq_residual, dtype=float).reshape(-1)
    eq_rows = int(r_eq.size)
    ineq_rows = int(g_in.size)

    if eq_jacobian is None:
        if eq_rows > 0:
            raise ValueError("check_kkt_residuals: eq_jacobian is required when eq_residual is non-empty.")
        J_eq = np.zeros((0, n_vars), dtype=float)
    else:
        J_eq = np.asarray(eq_jacobian, dtype=float)

    if ineq_jacobian is None:
        if ineq_rows > 0:
            raise ValueError("check_kkt_residuals: ineq_jacobian is required when ineq_residual is non-empty.")
        J_in = np.zeros((0, n_vars), dtype=float)
    else:
        J_in = np.asarray(ineq_jacobian, dtype=float)

    if J_eq.ndim != 2 or J_eq.shape != (eq_rows, n_vars):
        raise ValueError(
            "check_kkt_residuals: eq_jacobian shape mismatch. "
            f"Expected {(eq_rows, n_vars)}, got {J_eq.shape}."
        )
    if J_in.ndim != 2 or J_in.shape != (ineq_rows, n_vars):
        raise ValueError(
            "check_kkt_residuals: ineq_jacobian shape mismatch. "
            f"Expected {(ineq_rows, n_vars)}, got {J_in.shape}."
        )

    active_mask = np.asarray(g_in >= -active_tol_f, dtype=bool) if ineq_rows > 0 else np.zeros((0,), dtype=bool)
    active_idxs = np.flatnonzero(active_mask)

    A_eq_t = np.asarray(J_eq.T, dtype=float)
    A_act_t = (
        np.asarray(J_in[active_idxs, :].T, dtype=float)
        if active_idxs.size > 0
        else np.zeros((n_vars, 0), dtype=float)
    )

    lambda_eq = np.zeros((eq_rows,), dtype=float)
    mu_in = np.zeros((ineq_rows,), dtype=float)

    if eq_rows > 0 or active_idxs.size > 0:
        A_total = np.hstack([A_eq_t, A_act_t])
        if A_total.shape[1] > 0:
            nu_raw, *_ = np.linalg.lstsq(A_total, -grad_obj, rcond=None)
            nu_raw = np.asarray(nu_raw, dtype=float).reshape(-1)
            if eq_rows > 0:
                lambda_eq = nu_raw[:eq_rows]
            if active_idxs.size > 0:
                mu_act = np.maximum(nu_raw[eq_rows : eq_rows + int(active_idxs.size)], 0.0)
                if eq_rows > 0:
                    rhs = -(grad_obj + A_act_t @ mu_act)
                    lambda_eq, *_ = np.linalg.lstsq(A_eq_t, rhs, rcond=None)
                    lambda_eq = np.asarray(lambda_eq, dtype=float).reshape(-1)
                mu_in[active_idxs] = mu_act

    stationarity = np.asarray(
        grad_obj + A_eq_t @ lambda_eq + J_in.T @ mu_in,
        dtype=float,
    ).reshape(-1)

    stationarity_inf = float(np.max(np.abs(stationarity))) if stationarity.size > 0 else 0.0
    eq_violation_inf = float(np.max(np.abs(r_eq))) if r_eq.size > 0 else 0.0
    ineq_violation = np.maximum(g_in, 0.0)
    ineq_violation_inf = float(np.max(ineq_violation)) if ineq_violation.size > 0 else 0.0
    dual_violation = np.maximum(-mu_in, 0.0)
    dual_violation_inf = float(np.max(dual_violation)) if dual_violation.size > 0 else 0.0
    complementarity = np.asarray(mu_in * g_in, dtype=float).reshape(-1)
    complementarity_inf = float(np.max(np.abs(complementarity))) if complementarity.size > 0 else 0.0

    reasons: list[str] = []
    if stationarity_inf > stationarity_tol_f:
        reasons.append(
            f"stationarity_inf={stationarity_inf:.3e} > stationarity_tol={stationarity_tol_f:.3e}"
        )
    if eq_violation_inf > eq_tol_f:
        reasons.append(f"eq_violation_inf={eq_violation_inf:.3e} > eq_tol={eq_tol_f:.3e}")
    if ineq_violation_inf > ineq_tol_f:
        reasons.append(f"ineq_violation_inf={ineq_violation_inf:.3e} > ineq_tol={ineq_tol_f:.3e}")
    if complementarity_inf > complementarity_tol_f:
        reasons.append(
            f"complementarity_inf={complementarity_inf:.3e} > complementarity_tol={complementarity_tol_f:.3e}"
        )
    if dual_violation_inf > dual_tol_f:
        reasons.append(f"dual_violation_inf={dual_violation_inf:.3e} > dual_tol={dual_tol_f:.3e}")

    ok = len(reasons) == 0
    if ok:
        msg = (
            "ok "
            f"(stationarity={stationarity_inf:.3e}, eq={eq_violation_inf:.3e}, "
            f"ineq={ineq_violation_inf:.3e}, comp={complementarity_inf:.3e}, "
            f"dual={dual_violation_inf:.3e})"
        )
    else:
        msg = "; ".join(reasons)

    return KKTCheckResult(
        ok=ok,
        message=msg,
        stationarity_inf=stationarity_inf,
        eq_violation_inf=eq_violation_inf,
        ineq_violation_inf=ineq_violation_inf,
        complementarity_inf=complementarity_inf,
        dual_violation_inf=dual_violation_inf,
        n_vars=n_vars,
        n_objective_rows=n_obj_rows,
        n_eq_rows=eq_rows,
        n_ineq_rows=ineq_rows,
        n_active_ineq_rows=int(active_idxs.size),
        lambda_eq=lambda_eq.copy(),
        mu_ineq=mu_in.copy(),
    )


def check_kkt_conditions(
    runtime: NLSRuntime,
    *,
    required: Iterable[StateKey] | None = None,
    objective_term_indices: Iterable[int] | None = None,
    eq_term_indices: Iterable[int] | None = None,
    ineq_term_indices: Iterable[int] | None = None,
    ineq_sense: str = "<=",
    active_tol: float = 1e-8,
    stationarity_tol: float = 1e-6,
    eq_tol: float = 1e-8,
    ineq_tol: float = 1e-8,
    complementarity_tol: float = 1e-8,
    dual_tol: float = 1e-10,
) -> KKTCheckResult:
    """Check first-order KKT residuals at current `runtime.pack` iterate."""

    _sense_norm, ineq_scale = _inequality_sign_to_standardized(ineq_sense)

    n_terms = int(len(runtime.problem.terms))
    eq_auto = tuple(runtime.find_constraint_term_indices(kind="eq"))
    ineq_auto = tuple(runtime.find_constraint_term_indices(kind="ineq"))
    eq_idxs = eq_auto if eq_term_indices is None else _normalize_term_indices(
        n_terms=n_terms,
        term_indices=eq_term_indices,
        name="check_kkt_conditions.eq_term_indices",
    )
    ineq_idxs = ineq_auto if ineq_term_indices is None else _normalize_term_indices(
        n_terms=n_terms,
        term_indices=ineq_term_indices,
        name="check_kkt_conditions.ineq_term_indices",
    )
    obj_idxs = (
        tuple(i for i in range(n_terms) if i not in set(eq_idxs) and i not in set(ineq_idxs))
        if objective_term_indices is None
        else _normalize_term_indices(
            n_terms=n_terms,
            term_indices=objective_term_indices,
            name="check_kkt_conditions.objective_term_indices",
        )
    )

    eq_set = set(eq_idxs)
    ineq_set = set(ineq_idxs)
    obj_set = set(obj_idxs)
    overlap_eq_ineq = sorted(eq_set & ineq_set)
    if len(overlap_eq_ineq) > 0:
        raise ValueError(
            "check_kkt_conditions: eq/ineq term index overlap is not allowed. "
            f"Overlap={overlap_eq_ineq}."
        )
    overlap_obj_constraints = sorted(obj_set & (eq_set | ineq_set))
    if len(overlap_obj_constraints) > 0:
        raise ValueError(
            "check_kkt_conditions: objective_term_indices must exclude constraint terms. "
            f"Overlap={overlap_obj_constraints}."
        )

    req = runtime.required_list(required)
    r_obj, J_obj = runtime.linearize_stacked_terms(
        required=req,
        weighted=True,
        term_indices=obj_idxs,
    )
    r_eq, J_eq = runtime.linearize_stacked_terms(
        required=req,
        weighted=False,
        term_indices=eq_idxs,
    )
    r_in_raw, J_in_raw = runtime.linearize_stacked_terms(
        required=req,
        weighted=False,
        term_indices=ineq_idxs,
    )

    r_obj = np.asarray(r_obj, dtype=float).reshape(-1)
    J_obj = np.asarray(J_obj, dtype=float)
    r_eq = np.asarray(r_eq, dtype=float).reshape(-1)
    J_eq = np.asarray(J_eq, dtype=float)
    g_in = np.asarray(ineq_scale * np.asarray(r_in_raw, dtype=float).reshape(-1), dtype=float).reshape(-1)
    J_in = np.asarray(ineq_scale * np.asarray(J_in_raw, dtype=float), dtype=float)
    grad_obj = np.asarray(J_obj.T @ r_obj, dtype=float).reshape(-1)

    return check_kkt_residuals(
        grad_objective=grad_obj,
        eq_residual=r_eq,
        eq_jacobian=J_eq,
        ineq_residual=g_in,
        ineq_jacobian=J_in,
        n_objective_rows=int(r_obj.size),
        active_tol=active_tol,
        stationarity_tol=stationarity_tol,
        eq_tol=eq_tol,
        ineq_tol=ineq_tol,
        complementarity_tol=complementarity_tol,
        dual_tol=dual_tol,
    )


__all__ = [
    "KKTCheckResult",
    "check_kkt_residuals",
    "check_kkt_conditions",
]
