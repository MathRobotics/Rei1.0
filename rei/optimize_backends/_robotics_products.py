"""Robotics-specific batching plans, installed by backend compile adapters."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..core.expr.nodes import GetStateExpr, JointPowerSquaredExpr, StackExpr, TrajectoryVarExpr
from ..core.state_cache import StateKey
from ..core.state_schema import canonical_field_name
from ..optimize.term_products import StateTermProducts

Array = np.ndarray


def _same_trajectory_map(left: Any, right: Any) -> bool:
    """Compare trajectory maps without densifying B-spline operators."""
    if left is right:
        return True
    if any(not hasattr(value, attr) for value in (left, right) for attr in ("A", "b", "steps", "q_dim", "p_dim")):
        return False
    if (
        int(left.steps) != int(right.steps)
        or int(left.q_dim) != int(right.q_dim)
        or int(left.p_dim) != int(right.p_dim)
        or not np.array_equal(np.asarray(left.b), np.asarray(right.b))
    ):
        return False
    left_basis = getattr(left.A, "basis", None)
    right_basis = getattr(right.A, "basis", None)
    if left_basis is not None or right_basis is not None:
        return (
            left_basis is not None
            and right_basis is not None
            and int(getattr(left.A, "q_dim", -1)) == int(getattr(right.A, "q_dim", -2))
            and np.array_equal(np.asarray(left_basis), np.asarray(right_basis))
        )
    return np.array_equal(np.asarray(left.A), np.asarray(right.A))



class RoboticsTermProducts(StateTermProducts):
    def __init__(self, builder: Any):
        self.builder = builder




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
        power_candidates: list[dict[str, Any]] = []
        reference_vars: list[Any] | None = None
        derivative_maps = self.builder.trajectory_derivative_maps
        expected_qdot_map = derivative_maps.get(1) if isinstance(derivative_maps, dict) else None
        for idx, expr, _cost, _term_name, _r_raw, _use_size in terms:
            if not isinstance(expr, StackExpr) or len(expr.parts) < 2:
                continue
            if all(isinstance(part, JointPowerSquaredExpr) for part in expr.parts):
                parts = list(expr.parts)
                vars_list = list(expr.vars)
                compatible_power = expected_qdot_map is not None and len(vars_list) > 0
                for part in parts:
                    torque = part.torque
                    velocity = part.velocity
                    if not isinstance(torque, GetStateExpr) or not isinstance(velocity, TrajectoryVarExpr):
                        compatible_power = False
                        break
                    key = torque.key_value
                    if (
                        getattr(key, "dtype", None) != "dynamics"
                        or canonical_field_name(str(getattr(key, "field", ""))) != "torque"
                        or getattr(getattr(key, "owner", None), "owner_type", None) != "total_joint"
                        or velocity.k is None
                        or int(velocity.k) != int(key.k)
                        or not _same_trajectory_map(velocity.trajectory, expected_qdot_map)
                        or len(torque.vars) != len(vars_list)
                        or len(torque.key_jacs) != len(vars_list)
                        or len(velocity.vars) != len(vars_list)
                        or any(var is not ref for var, ref in zip(torque.vars, vars_list, strict=True))
                        or any(var is not ref for var, ref in zip(velocity.vars, vars_list, strict=True))
                    ):
                        compatible_power = False
                        break
                if compatible_power:
                    if reference_vars is None:
                        reference_vars = vars_list
                    elif len(reference_vars) != len(vars_list) or any(
                        var is not ref for var, ref in zip(vars_list, reference_vars, strict=True)
                    ):
                        compatible_power = False
                if compatible_power:
                    power_candidates.append({"idx": idx, "parts": parts, "vars": vars_list})
                continue
            if not all(isinstance(part, GetStateExpr) for part in expr.parts):
                continue
            parts = list(expr.parts)
            if not all(getattr(part.key_value, "dtype", None) == "dynamics" for part in parts):
                continue

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
        if (len(candidates) < minimum_candidates and len(power_candidates) == 0) or reference_vars is None:
            return {}, zero, set()

        try:
            requests_by_var: list[list[tuple[StateKey, StateKey, Array]]] = [[] for _ in reference_vars]
            power_requests_by_var: list[list[tuple[StateKey, StateKey, Array]]] = [
                [] for _ in reference_vars
            ]
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

            for candidate in power_candidates:
                idx = int(candidate["idx"])
                rhs_raw = np.asarray(raw_rhs_by_index[idx], dtype=float).reshape(-1)
                if rhs_raw.size != len(candidate["parts"]):
                    raise ValueError("squared-power residual VJP RHS size does not match its time steps")
                for var_index, _var in enumerate(reference_vars):
                    for part, power_rhs in zip(candidate["parts"], rhs_raw, strict=True):
                        torque = part.torque
                        power_requests_by_var[var_index].append(
                            (
                                torque.key_value,
                                torque.key_jacs[var_index],
                                np.asarray([power_rhs], dtype=float),
                            )
                        )

            if callable(fused_vjp):
                fused_grads = []
                for requests, power_requests in zip(
                    requests_by_var,
                    power_requests_by_var,
                    strict=True,
                ):
                    if power_requests:
                        value = fused_vjp(
                            requests,
                            squared_power_requests=power_requests,
                        )
                    else:
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
                        for candidate in [*candidates, *power_candidates]
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
