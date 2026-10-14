"""Optional execution plans for sharing expression derivative products."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np

from ..core.expr.nodes import GetStateExpr, StackExpr
from ..core.state_cache import StateKey

Array = np.ndarray


class TermProductPlan(Protocol):
    """Return accelerated results; omitted terms use normal expression evaluation."""

    def shared_jvp(self, runtime: Any, indices: Iterable[int], direction: np.ndarray) -> dict[int, np.ndarray]: ...

    def residual_vjp(
        self, runtime: Any, *, terms: list[Any], raw_rhs_by_index: dict[int, np.ndarray],
    ) -> tuple[dict[int, np.ndarray], np.ndarray, set[int]]: ...

    def gradient_contributions(
        self, runtime: Any, *, term_indices: Iterable[int],
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]: ...


class StateTermProducts:
    """Generic batching for compatible state-expression products."""

    def shared_jvp(self, runtime, indices, direction):
        """Send compatible dynamics stacks together so backends can share tangents."""
        batch = getattr(runtime.state, 'jacobian_mul_many', None)
        if not callable(batch):
            return {}
        requests, selections = [], []
        for idx in indices:
            expr, _ = runtime.problem.terms[idx]
            if not isinstance(expr, StackExpr) or not expr.parts:
                continue
            if not all(isinstance(p, GetStateExpr) and len(p.vars) == len(p.key_jacs) == 1 for p in expr.parts):
                continue
            start = len(requests)
            for part in expr.parts:
                lo, hi = runtime.pack.slices[part.vars[0].name]
                if hi-lo != part.vars[0].dim():
                    raise ValueError('Shared JVP: variable size mismatch.')
                requests.append((part.key_value, part.key_jacs[0], direction[lo:hi]))
            selections.append((idx, start, len(requests)))
        if len(selections) < 2:
            return {}
        try:
            values = batch(requests)
        except (AttributeError, NotImplementedError):
            return {}
        if len(values) != len(requests):
            raise ValueError('Shared JVP: wrong number of products.')
        values = [np.asarray(v, dtype=float) for v in values]
        if any(v.ndim != 1 for v in values):
            raise ValueError('Shared JVP: expected vector products.')
        return {idx: np.concatenate(values[start:stop])
                for idx, start, stop in selections}


    def residual_vjp(
        self, runtime,
        *,
        terms: Iterable[tuple[int, Any, Any, str, Array, int]],
        raw_rhs_by_index: dict[int, Array],
    ) -> tuple[dict[int, Array], Array, set[int]]:
        """Fuse compatible stacked dynamics VJPs for ``residual_vjp``.

        The state-cache operation is deliberately given every term/time
        request at once.  Kots can then use one heterogeneous RoboKots VJP
        call for fields such as torque and torque_d1.  RoboKots returns their
        summed VJP, which is exactly the result needed by ``residual_vjp``.
        The legacy per-request route remains for backends with that contract.
        """
        batch_vjp = getattr(runtime.state, "jacobian_transpose_mul_many", None) if runtime.state is not None else None
        fused_vjp = getattr(runtime.state, "jacobian_transpose_mul_many_fused", None) if runtime.state is not None else None
        zero = np.zeros((int(runtime.pack.n_total),), dtype=float)
        if not callable(batch_vjp) and not callable(fused_vjp):
            return {}, zero, set()

        candidates: list[dict[str, Any]] = []
        reference_vars: list[Any] | None = None
        for idx, expr, _cost, _term_name, _r_raw, _use_size in terms:
            if not isinstance(expr, StackExpr) or len(expr.parts) < 2:
                continue
            if not all(isinstance(part, GetStateExpr) for part in expr.parts):
                continue
            parts = list(expr.parts)

            vars_list = list(expr.vars)
            if len(vars_list) == 0 or any(
                len(part.vars) != len(vars_list)
                or len(part.key_jacs) != len(vars_list)
                or any(var is not ref for var, ref in zip(part.vars, vars_list, strict=True))
                for part in parts
            ):
                continue
            if reference_vars is None:
                reference_vars = vars_list
            elif len(reference_vars) != len(vars_list) or any(
                var is not ref for var, ref in zip(vars_list, reference_vars, strict=True)
            ):
                continue
            candidates.append({"idx": idx, "parts": parts, "vars": vars_list})

        # Summed VJP also avoids per-time parameter gradients for one stack.
        # Keep its existing batch route when the backend cannot return a sum.
        minimum_candidates = 1 if callable(fused_vjp) else 2
        if len(candidates) < minimum_candidates or reference_vars is None:
            return {}, zero, set()

        try:
            requests_by_var: list[list[tuple[StateKey, StateKey, Array]]] = [[] for _ in reference_vars]
            request_owners: list[list[tuple[dict[str, Any], int]]] = [[] for _ in reference_vars]
            for candidate in candidates:
                idx = int(candidate["idx"])
                rhs_raw = np.asarray(raw_rhs_by_index[idx], dtype=float).reshape(-1)
                offset = 0
                rhs_parts: list[Array] = []
                for part in candidate["parts"]:
                    value = np.asarray(part.eval_value(runtime.ctx), dtype=float).reshape(-1)
                    stop = offset + int(value.size)
                    rhs_part = rhs_raw[offset:stop]
                    if rhs_part.size != value.size:
                        raise ValueError("batched residual VJP RHS partition size mismatch")
                    rhs_parts.append(rhs_part)
                    offset = stop
                if offset != rhs_raw.size:
                    raise ValueError("batched residual VJP RHS size does not match stacked state values")

                candidate["grads"] = [np.zeros((var.dim(),), dtype=float) for var in reference_vars]
                for var_index, _var in enumerate(reference_vars):
                    for part, rhs_part in zip(candidate["parts"], rhs_parts, strict=True):
                        requests_by_var[var_index].append((part.key_value, part.key_jacs[var_index], rhs_part))
                        request_owners[var_index].append((candidate, var_index))

            if callable(fused_vjp):
                fused_grads = []
                for requests in requests_by_var:
                    value = fused_vjp(requests)
                    fused_grads.append(np.asarray(value, dtype=float).reshape(-1))
                return (
                    {},
                    runtime._assemble_global_gradient(
                        term_name="fused_dynamics_residual_vjp",
                        expr_vars=reference_vars,
                        gradients=fused_grads,
                    ),
                    {
                        int(candidate["idx"])
                        for candidate in candidates
                    },
                )

            if not callable(batch_vjp):
                return {}, zero, set()

            for var_index, requests in enumerate(requests_by_var):
                contributions = [np.asarray(value, dtype=float) for value in batch_vjp(requests)]
                if len(contributions) != len(requests):
                    raise ValueError("batched residual VJP returned an unexpected number of contributions")
                for contribution, (candidate, owner_var_index) in zip(
                    contributions,
                    request_owners[var_index],
                    strict=True,
                ):
                    candidate["grads"][owner_var_index] += np.asarray(contribution, dtype=float).reshape(-1)

            return {
                int(candidate["idx"]): runtime._assemble_global_gradient(
                    term_name=runtime._term_display_name(int(candidate["idx"])),
                    expr_vars=candidate["vars"],
                    gradients=candidate["grads"],
                )
                for candidate in candidates
            }, zero, set()
        except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
            # Preserve the established per-expression VJP implementation for
            # unsupported state families and older backend contracts.
            return {}, zero, set()


    def gradient_contributions(
        self, runtime,
        *,
        term_indices: Iterable[int],
    ) -> dict[int, tuple[Array, Array, Array]]:
        """Evaluate compatible dynamics stacks across IOC terms in one VJP batch.

        RoboKots sums its heterogeneous VJP requests.  Matrix RHS columns
        preserve the individual term gradients while still using one reverse
        pass; the legacy per-request API remains the fallback.
        """
        batch_vjp = getattr(runtime.state, "jacobian_transpose_mul_many", None) if runtime.state is not None else None
        fused_column_vjp = (
            getattr(runtime.state, "jacobian_transpose_mul_many_fused_columns", None)
            if runtime.state is not None else None
        )
        if not callable(batch_vjp) and not callable(fused_column_vjp):
            return {}

        candidates: list[dict[str, Any]] = []
        value_required: list[StateKey] = []
        reference_vars: list[Any] | None = None
        for idx in term_indices:
            expr, cost = runtime.problem.terms[idx]
            if not isinstance(expr, StackExpr) or len(expr.parts) < 2:
                continue
            if not all(isinstance(part, GetStateExpr) for part in expr.parts):
                continue
            parts = list(expr.parts)
            vars_list = list(expr.vars)
            if len(vars_list) == 0 or any(
                len(part.vars) != len(vars_list)
                or len(part.key_jacs) != len(vars_list)
                or any(var is not ref for var, ref in zip(part.vars, vars_list, strict=True))
                for part in parts
            ):
                continue
            if reference_vars is None:
                reference_vars = vars_list
            elif len(reference_vars) != len(vars_list) or any(
                var is not ref for var, ref in zip(vars_list, reference_vars, strict=True)
            ):
                continue
            value_required.extend(part.key_value for part in parts)
            candidates.append({"idx": idx, "expr": expr, "cost": cost, "parts": parts, "vars": vars_list})

        if len(candidates) < 2 or reference_vars is None:
            return {}

        try:
            runtime.update_state_if_needed(required=list(dict.fromkeys(value_required)))
            requests_by_var: list[list[tuple[StateKey, StateKey, Array]]] = [[] for _ in reference_vars]
            objective_requests_by_var: list[list[list[tuple[StateKey, StateKey, Array]]]] = [
                [] for _ in reference_vars
            ]
            request_owners: list[list[tuple[dict[str, Any], int]]] = [[] for _ in reference_vars]
            for candidate in candidates:
                rhs_parts: list[Array] = []
                values = [np.asarray(part.eval_value(runtime.ctx), dtype=float).reshape(-1) for part in candidate["parts"]]
                raw = np.concatenate(values, axis=0)
                candidate["raw"] = raw
                offset = 0
                for value in values:
                    stop = offset + int(value.size)
                    rhs_parts.append(raw[offset:stop])
                    offset = stop
                candidate["grads"] = [np.zeros((var.dim(),), dtype=float) for var in reference_vars]
                for var_index, _var in enumerate(reference_vars):
                    objective_requests: list[tuple[StateKey, StateKey, Array]] = []
                    for part, rhs_part in zip(candidate["parts"], rhs_parts, strict=True):
                        request = (part.key_value, part.key_jacs[var_index], rhs_part)
                        requests_by_var[var_index].append(request)
                        objective_requests.append(request)
                        request_owners[var_index].append((candidate, var_index))
                    objective_requests_by_var[var_index].append(objective_requests)

            if callable(fused_column_vjp):
                for var_index, request_groups in enumerate(objective_requests_by_var):
                    columns = np.asarray(fused_column_vjp(request_groups), dtype=float)
                    expected = (reference_vars[var_index].dim(), len(candidates))
                    if columns.shape != expected:
                        raise ValueError(
                            "fused-column residual VJP shape mismatch. "
                            f"Expected {expected}, got {columns.shape}."
                        )
                    for column, candidate in enumerate(candidates):
                        candidate["grads"][var_index] += columns[:, column]
                out: dict[int, tuple[Array, Array, Array]] = {}
                for candidate in candidates:
                    raw = np.asarray(candidate["raw"], dtype=float).reshape(-1)
                    zeros = [np.zeros((raw.size, var.dim()), dtype=float) for var in candidate["vars"]]
                    apply_cost = getattr(candidate["cost"], "apply", None)
                    weighted = (np.asarray(apply_cost(raw, zeros)[0], dtype=float).reshape(-1)
                                if callable(apply_cost) else raw.copy())
                    out[int(candidate["idx"])] = (
                        raw, weighted,
                        runtime._assemble_global_gradient(
                            term_name=runtime._term_display_name(int(candidate["idx"])),
                            expr_vars=candidate["vars"], gradients=candidate["grads"],
                        ),
                    )
                return out

            for var_index, requests in enumerate(requests_by_var):
                contributions = [np.asarray(value, dtype=float) for value in batch_vjp(requests)]
                if len(contributions) != len(requests):
                    raise ValueError("batched IOC VJP returned an unexpected number of contributions")
                for contribution, (candidate, owner_var_index) in zip(
                    contributions,
                    request_owners[var_index],
                    strict=True,
                ):
                    candidate["grads"][owner_var_index] += np.asarray(contribution, dtype=float).reshape(-1)

            out: dict[int, tuple[Array, Array, Array]] = {}
            for candidate in candidates:
                raw = np.asarray(candidate["raw"], dtype=float).reshape(-1)
                zeros = [np.zeros((raw.size, var.dim()), dtype=float) for var in candidate["vars"]]
                apply_cost = getattr(candidate["cost"], "apply", None)
                if callable(apply_cost):
                    weighted, _blocks = apply_cost(raw, zeros)
                    weighted = np.asarray(weighted, dtype=float).reshape(-1)
                else:
                    weighted = raw.copy()
                out[int(candidate["idx"])] = (
                    raw,
                    weighted,
                    runtime._assemble_global_gradient(
                        term_name=runtime._term_display_name(int(candidate["idx"])),
                        expr_vars=candidate["vars"],
                        gradients=candidate["grads"],
                    ),
                )
            return out
        except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
            return {}
