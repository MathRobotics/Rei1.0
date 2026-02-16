from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..backends.state.pinocchio import PinocchioTrajectoryStateBuilder
from ..backends.state.spatial import Jacobian6Order
from ..core.state_schema import DTYPE_DYNAMICS, canonical_dtype_name, canonical_field_name, split_jac_field
from ..core.trajectory import TrajectoryMap
from ..optimize.dsl.dsl_ops import iter_nodes
from ..optimize.dsl.trajectory_compile import PreparedTrajectoryProblemDsl
from ..optimize.runtime import NLSRuntime
from .trajectory_adapter import compile_trajectory_problem_with_adapter

Array = np.ndarray


@dataclass(frozen=True)
class PinocchioTrajectoryCompiledProblem:
    runtime: NLSRuntime
    trajectory_map: TrajectoryMap
    trajectory_derivative_maps: dict[int, TrajectoryMap]
    p_var: str
    dt: float
    model_order: int
    dynamics_fields: tuple[str, ...] = ()


def _infer_model_dof(model: Any) -> int | None:
    dof_fn = getattr(model, "dof", None)
    if callable(dof_fn):
        try:
            return int(dof_fn())
        except Exception:
            return None
    for attr in ("nq", "nv"):
        raw = getattr(model, attr, None)
        if raw is None:
            continue
        try:
            return int(raw)
        except Exception:
            continue
    return None


def _infer_model_order(model: Any) -> int:
    order_fn = getattr(model, "order", None)
    if callable(order_fn):
        try:
            return max(1, int(order_fn()))
        except Exception:
            pass
    order_attr = getattr(model, "order_", None)
    if order_attr is None:
        return 1
    try:
        return max(1, int(order_attr))
    except Exception:
        return 1


def _base_field_name(field: str) -> str:
    field_name = canonical_field_name(str(field))
    try:
        base, _var = split_jac_field(field_name)
    except ValueError:
        return field_name
    return base


def _canonicalize_dynamics_fields(
    dynamics_fields: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if dynamics_fields is None:
        return None
    out: list[str] = []
    seen: set[str] = set()
    for field_raw in dynamics_fields:
        field = canonical_field_name(str(field_raw).strip())
        if field == "":
            continue
        if field in seen:
            continue
        seen.add(field)
        out.append(field)
    if len(out) == 0:
        raise ValueError("compile_pinocchio_trajectory_problem: dynamics_fields must be non-empty when provided.")
    return tuple(out)


def _registered_dynamics_base_fields(
    *,
    builder: PinocchioTrajectoryStateBuilder,
    owner_type: str,
) -> set[str]:
    fields: set[str] = set()
    for field in builder.registered_route_fields(
        dtype=DTYPE_DYNAMICS,
        owner_type=owner_type,
    ):
        fields.add(_base_field_name(field))
    return fields


def _required_dynamics_base_fields_in_order(
    *,
    runtime: NLSRuntime,
    owner_type: str,
) -> tuple[list[str], set[str]]:
    requested_fields: list[str] = []
    requested_seen: set[str] = set()
    unsupported_owner_types: set[str] = set()
    for key in runtime.required:
        if getattr(key, "dtype", None) != DTYPE_DYNAMICS:
            continue
        owner = getattr(key, "owner", None)
        key_owner_type = getattr(owner, "owner_type", None)
        if key_owner_type != owner_type:
            unsupported_owner_types.add(str(key_owner_type))
            continue
        field = _base_field_name(str(getattr(key, "field", "")))
        if field in requested_seen:
            continue
        requested_seen.add(field)
        requested_fields.append(field)
    return requested_fields, unsupported_owner_types


def _required_dynamics_base_fields(
    *,
    runtime: NLSRuntime,
    owner_type: str,
) -> tuple[set[str], set[str]]:
    requested_fields, unsupported_owner_types = _required_dynamics_base_fields_in_order(
        runtime=runtime,
        owner_type=owner_type,
    )
    return set(requested_fields), unsupported_owner_types


def _explicit_get_state_jac_field_bases(node: Mapping[str, Any]) -> list[str]:
    jac_entries: list[Mapping[str, Any]] = []

    jac_dsl = node.get("jac", None)
    if jac_dsl is not None:
        if not isinstance(jac_dsl, Mapping):
            raise ValueError("get_state: jac must be a dict when provided.")
        jac_entries.append(jac_dsl)

    jacs_dsl = node.get("jacs", None)
    if jacs_dsl is not None:
        if not isinstance(jacs_dsl, list):
            raise ValueError("get_state: jacs must be a list[dict] when provided.")
        for i, item in enumerate(jacs_dsl):
            if not isinstance(item, Mapping):
                raise ValueError(f"get_state: jacs[{i}] must be a dict, got {type(item).__name__}.")
            jac_entries.append(item)

    out: list[str] = []
    for entry in jac_entries:
        field_raw = entry.get("field", None)
        if field_raw is None:
            continue
        out.append(_base_field_name(str(field_raw)))
    return out


def _required_dynamics_base_fields_in_order_from_dsl(
    *,
    dsl: Mapping[str, Any],
    owner_type: str,
) -> tuple[list[str], set[str]]:
    requested_fields: list[str] = []
    requested_seen: set[str] = set()
    unsupported_owner_types: set[str] = set()

    for term in dsl.get("terms", []) or []:
        if not isinstance(term, Mapping):
            continue
        expr = term.get("expr", None)
        for node in iter_nodes(expr):
            if node.get("type", None) != "get_state":
                continue
            key = node.get("key", None)
            if not isinstance(key, Mapping):
                continue

            dtype_name = canonical_dtype_name(str(key.get("dtype", "")))
            if dtype_name != DTYPE_DYNAMICS:
                continue

            key_owner_type = str(key.get("owner_type", ""))
            if key_owner_type != owner_type:
                unsupported_owner_types.add(key_owner_type)
                continue

            field = _base_field_name(str(key.get("field", "")))
            if field not in requested_seen:
                requested_seen.add(field)
                requested_fields.append(field)

            for jac_field_base in _explicit_get_state_jac_field_bases(node):
                if jac_field_base in requested_seen:
                    continue
                requested_seen.add(jac_field_base)
                requested_fields.append(jac_field_base)

    return requested_fields, unsupported_owner_types


def _validate_pinocchio_runtime_dynamics_coverage(
    *,
    runtime: NLSRuntime,
    builder: PinocchioTrajectoryStateBuilder,
    dynamics_owner_type: str,
) -> None:
    requested_fields, unsupported_owner_types = _required_dynamics_base_fields(
        runtime=runtime,
        owner_type=dynamics_owner_type,
    )
    if unsupported_owner_types:
        unsupported = ", ".join(sorted(unsupported_owner_types))
        raise ValueError(
            "compile_pinocchio_trajectory_problem: DSL contains dynamics keys with unsupported owner_type(s): "
            f"{unsupported}. Supported owner_type is {dynamics_owner_type!r}."
        )
    if len(requested_fields) == 0:
        return

    registered_fields = _registered_dynamics_base_fields(
        builder=builder,
        owner_type=dynamics_owner_type,
    )
    missing_fields = sorted(requested_fields - registered_fields)
    if len(missing_fields) == 0:
        return

    requested_str = ", ".join(sorted(requested_fields))
    registered_str = ", ".join(sorted(registered_fields)) if len(registered_fields) > 0 else "<none>"
    missing_str = ", ".join(missing_fields)
    raise ValueError(
        "compile_pinocchio_trajectory_problem: DSL requests dynamics field(s) that are not registered in "
        "PinocchioTrajectoryStateBuilder. "
        f"Missing: {missing_str}. Requested: {requested_str}. Registered: {registered_str}. "
        "Add missing entries to `dynamics_fields` (or dynamics_custom_handlers), "
        "or remove corresponding get_state dynamics terms."
    )


@dataclass
class _PinocchioTrajectoryCompileAdapter:
    jac6_order: Jacobian6Order = "linear_angular"
    finite_diff_eps: float = 1e-8
    fields: Sequence[str] | None = None
    dynamics_fields: Sequence[str] | None = None
    dynamics_owner_type: str = "total_joint"
    dynamics_custom_handlers: Mapping[str, tuple[Callable[..., Array], Callable[..., Array]]] | None = None
    resolved_dynamics_fields: tuple[str, ...] = ()

    def infer_model_dof(self, model: Any) -> int | None:
        return _infer_model_dof(model)

    def infer_model_order(self, model: Any) -> int:
        return _infer_model_order(model)

    def _resolve_dynamics_fields(
        self,
        *,
        model: Any,
        data: Any,
        prepared: PreparedTrajectoryProblemDsl,
    ) -> tuple[str, ...] | None:
        del model, data
        dynamics_fields_use = _canonicalize_dynamics_fields(self.dynamics_fields)
        if dynamics_fields_use is None:
            requested_fields_order, unsupported_owner_types = _required_dynamics_base_fields_in_order_from_dsl(
                dsl=prepared.dsl,
                owner_type=self.dynamics_owner_type,
            )
            if unsupported_owner_types:
                unsupported = ", ".join(sorted(unsupported_owner_types))
                raise ValueError(
                    "compile_pinocchio_trajectory_problem: DSL contains dynamics keys with unsupported owner_type(s): "
                    f"{unsupported}. Supported owner_type is {self.dynamics_owner_type!r}."
                )
            dynamics_fields_use = tuple(requested_fields_order) if len(requested_fields_order) > 0 else None
        return dynamics_fields_use

    def build_state_builder(
        self,
        *,
        model: Any,
        data: Any,
        prepared: PreparedTrajectoryProblemDsl,
    ) -> PinocchioTrajectoryStateBuilder:
        dynamics_fields_use = self._resolve_dynamics_fields(
            model=model,
            data=data,
            prepared=prepared,
        )
        self.resolved_dynamics_fields = (
            tuple() if dynamics_fields_use is None else tuple(dynamics_fields_use)
        )
        return PinocchioTrajectoryStateBuilder(
            model,
            data,
            trajectory_map=prepared.trajectory_map,
            trajectory_derivative_maps=prepared.trajectory_derivative_maps,
            p_var=prepared.p_var,
            jac6_order=self.jac6_order,
            finite_diff_eps=self.finite_diff_eps,
            fields=self.fields,
            dynamics_fields=dynamics_fields_use,
            dynamics_owner_type=self.dynamics_owner_type,
            dynamics_custom_handlers=self.dynamics_custom_handlers,
        )

    def validate_runtime(
        self,
        *,
        runtime: NLSRuntime,
        state_builder: PinocchioTrajectoryStateBuilder,
        prepared: PreparedTrajectoryProblemDsl,
    ) -> None:
        del prepared
        _validate_pinocchio_runtime_dynamics_coverage(
            runtime=runtime,
            builder=state_builder,
            dynamics_owner_type=self.dynamics_owner_type,
        )


def compile_pinocchio_trajectory_problem(
    dsl: Mapping[str, Any],
    *,
    model: Any,
    data: Any,
    p_var: str | None = None,
    max_derivative_order: int | None = None,
    derivative_wrt: str = "time",
    default_steps: int | None = None,
    default_q_dim: int | None = None,
    default_dt: float | None = None,
    jac6_order: Jacobian6Order = "linear_angular",
    finite_diff_eps: float = 1e-8,
    fields: Sequence[str] | None = None,
    dynamics_fields: Sequence[str] | None = None,
    dynamics_owner_type: str = "total_joint",
    dynamics_custom_handlers: Mapping[str, tuple[Callable[..., Array], Callable[..., Array]]] | None = None,
) -> PinocchioTrajectoryCompiledProblem:
    adapter = _PinocchioTrajectoryCompileAdapter(
        jac6_order=jac6_order,
        finite_diff_eps=finite_diff_eps,
        fields=fields,
        dynamics_fields=dynamics_fields,
        dynamics_owner_type=dynamics_owner_type,
        dynamics_custom_handlers=dynamics_custom_handlers,
    )
    compiled = compile_trajectory_problem_with_adapter(
        dsl,
        model=model,
        data=data,
        adapter=adapter,
        p_var=p_var,
        max_derivative_order=max_derivative_order,
        derivative_wrt=derivative_wrt,
        default_steps=default_steps,
        default_q_dim=default_q_dim,
        default_dt=default_dt,
    )
    return PinocchioTrajectoryCompiledProblem(
        runtime=compiled.runtime,
        trajectory_map=compiled.prepared.trajectory_map,
        trajectory_derivative_maps=compiled.prepared.trajectory_derivative_maps,
        p_var=compiled.prepared.p_var,
        dt=float(compiled.prepared.dt),
        model_order=int(compiled.prepared.model_order),
        dynamics_fields=tuple(adapter.resolved_dynamics_fields),
    )


__all__ = [
    "PinocchioTrajectoryCompiledProblem",
    "compile_pinocchio_trajectory_problem",
]
