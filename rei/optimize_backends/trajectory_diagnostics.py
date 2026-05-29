from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Literal

from ..core.state_schema import DTYPE_DYNAMICS, canonical_dtype_name, canonical_field_name, torque_derivative_order
from ..optimize.dsl.dsl_ops import iter_nodes
from ._state_field_utils import base_field_name

UnsupportedPolicy = Literal["error", "warn_skip"]


@dataclass(frozen=True)
class BackendFieldCapability:
    dtype: str
    owner_type: str
    field: str
    value: bool
    jacobian_wrt_p: bool
    required_derivative_order: int | None
    required_model_order: int | None
    reason: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnsupportedTermDiagnostic:
    term_index: int
    term_name: str
    dtype: str
    field: str
    backend: str
    reason: str
    action: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryProblemDiagnostics:
    backend: str
    requested_terms: tuple[dict[str, Any], ...] = ()
    supported_terms: tuple[int, ...] = ()
    unsupported_terms: tuple[UnsupportedTermDiagnostic, ...] = ()
    warnings: tuple[UnsupportedTermDiagnostic, ...] = ()
    capabilities: tuple[BackendFieldCapability, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "requested_terms": [dict(t) for t in self.requested_terms],
            "supported_terms": [int(i) for i in self.supported_terms],
            "unsupported_terms": [d.to_json_dict() for d in self.unsupported_terms],
            "warnings": [d.to_json_dict() for d in self.warnings],
            "capabilities": [c.to_json_dict() for c in self.capabilities],
        }


def normalize_unsupported_policy(unsupported: str) -> UnsupportedPolicy:
    value = str(unsupported).strip().lower()
    if value in ("error", "strict"):
        return "error"
    if value in ("warn_skip", "skip", "warning_skip"):
        return "warn_skip"
    raise ValueError("unsupported must be one of 'error' or 'warn_skip'.")


def _term_name(term: Mapping[str, Any], *, fallback: str) -> str:
    name = term.get("name", None)
    if name is not None and str(name).strip() != "":
        return str(name)
    expr = term.get("expr", None)
    if isinstance(expr, Mapping):
        expr_name = expr.get("name", None)
        if expr_name is not None and str(expr_name).strip() != "":
            return str(expr_name)
    return fallback


def _trajectory_derivative_order_from_get_traj_var(node: Mapping[str, Any]) -> int | None:
    if node.get("type", None) != "get_traj_var":
        return None
    max_order = node.get(
        "max_derivative_order",
        node.get("derivative_order_max", node.get("max_deriv_order", None)),
    )
    if max_order is not None:
        return int(max_order)
    return int(node.get("derivative_order", node.get("deriv_order", 0)))


def _dynamics_requirement(field: str) -> tuple[int | None, int | None]:
    deriv = torque_derivative_order(field)
    if deriv is None:
        return None, None
    required_derivative_order = 2 + int(deriv)
    required_model_order = None if deriv <= 0 else 3 + int(deriv)
    return required_derivative_order, required_model_order


def _supported_dynamics_fields(
    *,
    backend: str,
    extra_supported_dynamics_fields: Sequence[str] | None = None,
) -> set[str]:
    backend_name = str(backend).strip().lower()
    extra = {
        canonical_field_name(str(f))
        for f in (extra_supported_dynamics_fields or ())
        if str(f).strip() != ""
    }
    if backend_name == "pinocchio":
        return {"torque", "momentum", "force"} | extra
    if backend_name in ("kots", "robokots"):
        return {"momentum", "force", "torque", "torque_d1", "torque_d2", "torque_d3"} | extra
    return set(extra)


def inspect_trajectory_problem_backend(
    dsl: Mapping[str, Any],
    *,
    backend: str,
    model: Any | None = None,
    data: Any | None = None,
    model_order: int | None = None,
    max_derivative_order: int | None = None,
    dynamics_owner_type: str = "total_joint",
    extra_supported_dynamics_fields: Sequence[str] | None = None,
    unsupported_action: str = "error",
) -> TrajectoryProblemDiagnostics:
    """Inspect trajectory DSL requirements against a backend capability surface."""

    del data
    backend_name = str(backend).strip().lower()
    supported_dyn = _supported_dynamics_fields(
        backend=backend_name,
        extra_supported_dynamics_fields=extra_supported_dynamics_fields,
    )
    max_deriv = None if max_derivative_order is None else int(max_derivative_order)
    if model_order is None:
        order_fn = getattr(model, "order", None)
        if callable(order_fn):
            try:
                model_order = int(order_fn())
            except Exception:
                model_order = None
        if model_order is None:
            model_order = getattr(model, "order_", None)
    model_order_i = 3 if model_order is None else int(model_order)
    unsupported_diags: list[UnsupportedTermDiagnostic] = []
    capabilities: list[BackendFieldCapability] = []
    requested_terms: list[dict[str, Any]] = []
    supported_term_indices: list[int] = []

    terms = dsl.get("terms", []) or []
    for term_index, term_raw in enumerate(terms):
        if not isinstance(term_raw, Mapping):
            continue
        term_name = _term_name(term_raw, fallback=f"term_{term_index}")
        term_supported = True
        term_requests: list[dict[str, Any]] = []
        expr = term_raw.get("expr", None)
        for node in iter_nodes(expr):
            if node.get("type", None) == "get_traj_var":
                order = _trajectory_derivative_order_from_get_traj_var(node)
                if order is None:
                    continue
                field = "q" if int(order) == 0 else f"q{'d' * int(order)}ot"
                term_requests.append(
                    {
                        "kind": "trajectory_derivative",
                        "field": field,
                        "required_derivative_order": int(order),
                    }
                )
                supported = max_deriv is None or int(order) <= int(max_deriv)
                capabilities.append(
                    BackendFieldCapability(
                        dtype="trajectory",
                        owner_type="total_joint",
                        field=field,
                        value=supported,
                        jacobian_wrt_p=supported,
                        required_derivative_order=int(order),
                        required_model_order=None,
                        reason=None
                        if supported
                        else (
                            f"trajectory derivative order {int(order)} requires "
                            f"max_derivative_order >= {int(order)}"
                        ),
                    )
                )
                if not supported:
                    term_supported = False
                    unsupported_diags.append(
                        UnsupportedTermDiagnostic(
                            term_index=int(term_index),
                            term_name=term_name,
                            dtype="trajectory",
                            field=field,
                            backend=backend_name,
                            reason=(
                                f"trajectory derivative order {int(order)} is unavailable; "
                                f"max_derivative_order={max_deriv}"
                            ),
                            action=str(unsupported_action),
                        )
                    )
                continue

            if node.get("type", None) != "get_state":
                continue
            key = node.get("key", None)
            if not isinstance(key, Mapping):
                continue
            dtype = canonical_dtype_name(str(key.get("dtype", "")))
            if dtype != DTYPE_DYNAMICS:
                continue
            owner_type = str(key.get("owner_type", ""))
            field = base_field_name(str(key.get("field", "")))
            req_deriv, req_model_order = _dynamics_requirement(field)
            term_requests.append(
                {
                    "kind": "dynamics",
                    "dtype": dtype,
                    "owner_type": owner_type,
                    "field": field,
                    "required_derivative_order": req_deriv,
                    "required_model_order": req_model_order,
                }
            )
            supported = True
            reason = None
            if owner_type != str(dynamics_owner_type):
                supported = False
                reason = (
                    f"dynamics owner_type {owner_type!r} is not supported by this compile path; "
                    f"expected {dynamics_owner_type!r}"
                )
            elif field not in supported_dyn:
                supported = False
                reason = f"dynamics field is not implemented by {backend_name} trajectory state builder"
            elif req_deriv is not None and max_deriv is not None and int(req_deriv) > int(max_deriv):
                supported = False
                reason = (
                    f"dynamics field {field!r} requires trajectory derivative order "
                    f"{int(req_deriv)}, max_derivative_order={int(max_deriv)}"
                )
            elif req_model_order is not None and model_order_i < int(req_model_order):
                supported = False
                reason = (
                    f"dynamics field {field!r} requires model_order >= {int(req_model_order)}, "
                    f"model_order={model_order_i}"
                )
            capabilities.append(
                BackendFieldCapability(
                    dtype=dtype,
                    owner_type=owner_type,
                    field=field,
                    value=supported,
                    jacobian_wrt_p=supported,
                    required_derivative_order=req_deriv,
                    required_model_order=req_model_order,
                    reason=reason,
                )
            )
            if not supported:
                term_supported = False
                unsupported_diags.append(
                    UnsupportedTermDiagnostic(
                        term_index=int(term_index),
                        term_name=term_name,
                        dtype=dtype,
                        field=field,
                        backend=backend_name,
                        reason=str(reason),
                        action=str(unsupported_action),
                    )
                )
        requested_terms.append(
            {
                "term_index": int(term_index),
                "term_name": term_name,
                "requests": term_requests,
            }
        )
        if term_supported:
            supported_term_indices.append(int(term_index))

    return TrajectoryProblemDiagnostics(
        backend=backend_name,
        requested_terms=tuple(requested_terms),
        supported_terms=tuple(supported_term_indices),
        unsupported_terms=tuple(unsupported_diags),
        warnings=tuple(unsupported_diags) if str(unsupported_action) == "skipped" else (),
        capabilities=tuple(capabilities),
    )


def filter_unsupported_terms_from_dsl(
    dsl: Mapping[str, Any],
    diagnostics: TrajectoryProblemDiagnostics,
) -> dict[str, Any]:
    out = deepcopy(dict(dsl))
    skip_indices = {int(d.term_index) for d in diagnostics.unsupported_terms}
    terms = out.get("terms", []) or []
    out["terms"] = [term for i, term in enumerate(terms) if int(i) not in skip_indices]
    return out


def raise_for_unsupported_terms(
    diagnostics: TrajectoryProblemDiagnostics,
    *,
    error_prefix: str,
) -> None:
    if len(diagnostics.unsupported_terms) == 0:
        return
    parts = [
        f"term[{d.term_index}] {d.term_name!r} field={d.field!r}: {d.reason}"
        for d in diagnostics.unsupported_terms
    ]
    raise ValueError(f"{error_prefix}: unsupported trajectory problem term(s): " + "; ".join(parts))


__all__ = [
    "BackendFieldCapability",
    "TrajectoryProblemDiagnostics",
    "UnsupportedPolicy",
    "UnsupportedTermDiagnostic",
    "filter_unsupported_terms_from_dsl",
    "inspect_trajectory_problem_backend",
    "normalize_unsupported_policy",
    "raise_for_unsupported_terms",
]
