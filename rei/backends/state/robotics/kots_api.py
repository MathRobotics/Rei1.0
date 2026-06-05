from __future__ import annotations

import re
from typing import Any

import numpy as np

from ....core.state_schema import torque_derivative_order
from ..jacobian_ops import as_2d_numeric_matrix, normalize_jvp_output
from ...optional import import_optional_backend, require_module_attrs

Array = np.ndarray
ROBOKOTS_TORQUE_DIFF_PATTERN = re.compile(r"^torque_diff([1-9][0-9]*)$")
ROBOKOTS_JACOBIAN_MUL_METHODS = ("jacobian_mul", "matvec", "jacobian_matvec")

_state_mod = import_optional_backend(
    "robokots.core.state",
    backend_name="rei.backends.state.robotics.kots",
    install_hint="uv sync --group kots",
)
require_module_attrs(
    _state_mod,
    ("StateType",),
    backend_name="rei.backends.state.robotics.kots",
    install_hint="uv sync --group kots",
)
StateType = _state_mod.StateType


def fallback_backend_field_name(state_field: str) -> str:
    deriv_order = torque_derivative_order(state_field)
    if isinstance(deriv_order, int) and deriv_order > 0:
        return f"torque_diff{deriv_order}"
    return state_field


def make_state_type(
    *,
    owner_type: str,
    owner_name: str,
    state_field: str,
    frame_name: str | None,
    state_type: Any = StateType,
) -> Any:
    try:
        return state_type(owner_type, owner_name, state_field, frame_name)
    except KeyError:
        fallback_field = fallback_backend_field_name(state_field)
        if fallback_field == state_field:
            raise
        return state_type(owner_type, owner_name, fallback_field, frame_name)


def torque_derivative_order_from_state_data_type(data_type: str) -> int | None:
    try:
        deriv_order = torque_derivative_order(data_type)
    except ValueError:
        deriv_order = None
    if isinstance(deriv_order, int):
        return deriv_order
    m = ROBOKOTS_TORQUE_DIFF_PATTERN.fullmatch(data_type)
    if m is None:
        return None
    return int(m.group(1))


def state_ref_data_type(state_ref: Any) -> str | None:
    for attr in ("data_type", "field", "field_", "dtype"):
        value = getattr(state_ref, attr, None)
        if isinstance(value, str) and value != "":
            return value
    return None


def state_ref_owner_type(state_ref: Any) -> str | None:
    value = getattr(state_ref, "owner_type", None)
    if isinstance(value, str) and value != "":
        return value
    return None


def state_ref_owner_name(state_ref: Any) -> str | None:
    value = getattr(state_ref, "owner_name", None)
    if isinstance(value, str) and value != "":
        return value
    return None


def state_ref_frame_name(state_ref: Any) -> str | None:
    for attr in ("frame", "frame_name"):
        value = getattr(state_ref, attr, None)
        if isinstance(value, str) and value != "":
            return value
    return None


def state_info(model: Any, state_ref: Any) -> Array:
    return np.asarray(model.state_info(state_ref), dtype=float).reshape(-1)


def state_info_list(model: Any, refs: tuple[Any, ...]) -> Array | None:
    fn = getattr(model, "state_info_list", None)
    if not callable(fn):
        return None
    try:
        return np.asarray(fn(list(refs)), dtype=float).reshape(-1)
    except (KeyError, ValueError, TypeError, RuntimeError):
        return None


def jacobian(model: Any, state_ref: Any) -> Array:
    return np.asarray(model.jacobian(state_ref), dtype=float)


def jacobian_list(model: Any, refs: tuple[Any, ...]) -> Array | None:
    try:
        return np.asarray(model.jacobian(list(refs)), dtype=float)
    except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
        return None


def _candidate_jacobian_mul_methods(model: Any) -> tuple[Any, ...]:
    methods = []
    for name in ROBOKOTS_JACOBIAN_MUL_METHODS:
        fn = getattr(model, name, None)
        if callable(fn):
            methods.append(fn)
    return tuple(methods)


def _call_with_ref_arg_orders(fn: Any, state_ref: Any, arg: Array) -> Array:
    errors: list[Exception] = []
    for args in ((state_ref, arg), (arg, state_ref)):
        try:
            return np.asarray(fn(*args), dtype=float)
        except (KeyError, ValueError, TypeError, RuntimeError) as exc:
            errors.append(exc)
            continue
    if errors:
        raise errors[-1]
    raise AttributeError("RoboKots callable is not usable for state_ref/arg orders.")


def _fallback_jacobian_vec_mul(model: Any, state_ref: Any, vec: Array) -> Array:
    v = np.asarray(vec, dtype=float).reshape(-1)
    errors: list[Exception] = []
    for fn in _candidate_jacobian_mul_methods(model):
        try:
            return _call_with_ref_arg_orders(fn, state_ref, v).reshape(-1)
        except (KeyError, ValueError, TypeError, RuntimeError) as exc:
            errors.append(exc)
            continue
    if len(errors) > 0:
        raise errors[-1]
    raise AttributeError("RoboKots model does not expose jacobian_mul(state_ref, vec).")


def _fallback_jacobian_matrix_mul(model: Any, state_ref: Any, cols: Array) -> Array:
    C = as_2d_numeric_matrix(cols, name="RoboKots jacobian_mul columns")

    errors: list[Exception] = []
    fn = getattr(model, "jacobian_mul", None)
    if not callable(fn):
        raise AttributeError("RoboKots model does not expose jacobian_mul(state_ref, cols).")

    for args in ((state_ref, C), (C, state_ref)):
        try:
            return normalize_jvp_output(fn(*args), input_cols=C, backend_name="RoboKots")
        except (KeyError, ValueError, TypeError, RuntimeError) as exc:
            errors.append(exc)
            continue

    if len(errors) > 0:
        raise errors[-1]
    raise AttributeError("RoboKots model does not expose jacobian_mul(state_ref, cols).")


def _fallback_jacobian_from_vec_mul(model: Any, state_ref: Any, cols: Array, *, value_size: int) -> Array:
    C = as_2d_numeric_matrix(cols, name="RoboKots matvec columns")
    try:
        return _fallback_jacobian_matrix_mul(model, state_ref, C)
    except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
        pass

    return _assemble_jacobian_from_vector_mul(model, state_ref, C, value_size=value_size)


def _assemble_jacobian_from_vector_mul(model: Any, state_ref: Any, cols: Array, *, value_size: int | None) -> Array:
    C = as_2d_numeric_matrix(cols, name="RoboKots matvec columns")
    parts: list[Array] = []
    for j in range(int(C.shape[1])):
        parts.append(_fallback_jacobian_vec_mul(model, state_ref, C[:, j]))
    if len(parts) == 0:
        if value_size is None:
            raise ValueError("RoboKots cannot infer JVP output size from zero input columns.")
        return np.zeros((int(value_size), 0), dtype=float)
    m = int(parts[0].size)
    for part in parts[1:]:
        if int(part.size) != m:
            raise ValueError(
                "RoboKots inconsistent matvec output size while assembling Jacobian. "
                f"Expected {m}, got {part.size}."
            )
    return np.column_stack(parts)


def _fallback_jacobian_transpose_mul(model: Any, state_ref: Any, rhs: Array) -> Array:
    R = np.asarray(rhs, dtype=float)
    if R.ndim not in (1, 2):
        raise ValueError(f"RoboKots jacobian_transpose_mul rhs must be 1D or 2D, got shape {R.shape}.")

    errors: list[Exception] = []
    fn = getattr(model, "jacobian_transpose_mul", None)
    if not callable(fn):
        raise AttributeError("RoboKots model does not expose jacobian_transpose_mul(state_ref, rhs).")

    for args in ((state_ref, R), (R, state_ref)):
        try:
            out = np.asarray(fn(*args), dtype=float)
        except (KeyError, ValueError, TypeError, RuntimeError) as e:
            errors.append(e)
            continue
        if out.ndim not in (1, 2):
            errors.append(
                ValueError(f"RoboKots jacobian_transpose_mul output must be 1D or 2D, got shape {out.shape}.")
            )
            continue
        return out

    if len(errors) > 0:
        raise errors[-1]
    raise AttributeError("RoboKots model does not expose jacobian_transpose_mul(state_ref, rhs).")


class RoboKotsJacobianOperator:
    """RoboKots Jacobian capability facade.

    Public methods are the surface used by `KotsStateBuilder`; fallback probing
    stays private in this module.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def dense(self, state_ref: Any) -> Array:
        return jacobian(self.model, state_ref)

    def dense_list(self, refs: tuple[Any, ...]) -> Array | None:
        return jacobian_list(self.model, refs)

    def jvp(self, state_ref: Any, cols: Array, *, value_size: int | None = None) -> Array:
        C = np.asarray(cols, dtype=float)
        if C.ndim == 1:
            return _fallback_jacobian_vec_mul(self.model, state_ref, C)
        if value_size is None:
            try:
                return _fallback_jacobian_matrix_mul(self.model, state_ref, C)
            except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
                return _assemble_jacobian_from_vector_mul(self.model, state_ref, C, value_size=None)
        return _fallback_jacobian_from_vec_mul(self.model, state_ref, C, value_size=int(value_size))

    def vjp(self, state_ref: Any, rhs: Array) -> Array:
        return _fallback_jacobian_transpose_mul(self.model, state_ref, rhs)


def jacobian_vec_mul(model: Any, state_ref: Any, vec: Array) -> Array:
    return RoboKotsJacobianOperator(model).jvp(state_ref, vec)


def jacobian_matrix_mul(model: Any, state_ref: Any, cols: Array) -> Array:
    return RoboKotsJacobianOperator(model).jvp(state_ref, cols)


def jacobian_from_mul(model: Any, state_ref: Any, cols: Array, *, value_size: int) -> Array:
    return RoboKotsJacobianOperator(model).jvp(state_ref, cols, value_size=value_size)


def jacobian_transpose_mul(model: Any, state_ref: Any, rhs: Array) -> Array:
    return RoboKotsJacobianOperator(model).vjp(state_ref, rhs)


__all__ = [
    "ROBOKOTS_JACOBIAN_MUL_METHODS",
    "ROBOKOTS_TORQUE_DIFF_PATTERN",
    "RoboKotsJacobianOperator",
    "StateType",
    "fallback_backend_field_name",
    "jacobian",
    "jacobian_from_mul",
    "jacobian_list",
    "jacobian_matrix_mul",
    "jacobian_transpose_mul",
    "jacobian_vec_mul",
    "make_state_type",
    "state_info",
    "state_info_list",
    "state_ref_data_type",
    "state_ref_frame_name",
    "state_ref_owner_name",
    "state_ref_owner_type",
    "torque_derivative_order_from_state_data_type",
]
