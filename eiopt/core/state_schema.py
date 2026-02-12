from __future__ import annotations

import re
from typing import Tuple

import numpy as np

from .state_cache import OwnerKey, StateKey

# Reference (naming): RoboKots `robokots/core/state.py`

# Coordinate frames (when `StateKey.frame` is used)
FRAME_NAMES: Tuple[str, str] = ("world", "local")
DEFAULT_FRAME: str = "world"

# Common owner types (not enforced; backends may extend)
OWNER_TYPES: Tuple[str, ...] = ("joint", "link", "total_link", "total_joint", "total")

# Recommended dtype values (not enforced)
DTYPE_KINEMATICS = "kinematics"
DTYPE_DYNAMICS = "dynamics"
DTYPE_COORD = "coord"

DEFAULT_ROBOT_NAME: str = "robot"

# Minimal standard set for a backend-agnostic library.
#
# Values should be numeric arrays so they can be cached/serialized easily.
# Recommended shapes (3D):
#   - pos   : (3,)
#   - rot   : (9,) row-major flatten of 3x3 rotation matrix
#   - frame : (12,) = [pos(3), rot_flat(9)]
#   - q     : (nq,) joint angles
# Jacobian fields follow `"{field}_J_{var}"` and must match value dimension.
FRAME_FIELDS: Tuple[str, str, str] = ("pos", "rot", "frame")
JOINT_FIELDS: Tuple[str, ...] = ("q",)

# Kinematics-like fields (minimal set; backends may extend)
KIN_FIELDS: Tuple[str, ...] = FRAME_FIELDS

# Dynamics-like fields (minimal set; backends may extend)
MOMENTUM_FIELDS: Tuple[str, ...] = ("momentum",)
FORCE_FIELDS: Tuple[str, ...] = ("force",)
TORQUE_FIELDS: Tuple[str, ...] = ("torque", "torque_d1")

DYNAMICS_FIELDS: Tuple[str, ...] = MOMENTUM_FIELDS + FORCE_FIELDS + TORQUE_FIELDS

FIELD_ALIASES: dict[str, str] = {}

_TORQUE_D_PATTERN = re.compile(r"^torque_d([0-9]+)$")
_DEPRECATED_TORQUE_ALIAS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^tau$"),
    re.compile(r"^h$"),
    re.compile(r"^wrench$"),
    re.compile(r"^dtau(?:[1-9][0-9]*)?$"),
    re.compile(r"^tau_diff(?:[1-9][0-9]*)?$"),
    re.compile(r"^torque_rate(?:[1-9][0-9]*)?$"),
    re.compile(r"^torque_dot(?:[1-9][0-9]*)?$"),
    re.compile(r"^torque_diff[1-9][0-9]*$"),
)
_DEPRECATED_DTYPE_ALIASES: tuple[str, ...] = ("joint",)


def torque_derivative_field(order: int) -> str:
    order_i = int(order)
    if order_i <= 0:
        raise ValueError(f"torque_derivative_field: order must be >= 1, got {order_i}.")
    return f"torque_d{order_i}"


def _canonicalize_torque_field(field: str) -> str:
    if field == "torque":
        return "torque"

    match = _TORQUE_D_PATTERN.fullmatch(field)
    if match is not None:
        order = int(match.group(1))
        if order >= 1:
            return torque_derivative_field(order)
        return field

    return field


def _is_deprecated_torque_alias(field: str) -> bool:
    for pattern in _DEPRECATED_TORQUE_ALIAS_PATTERNS:
        if pattern.fullmatch(field) is not None:
            return True
    return False


def jac_field(field: str, *, var: str) -> str:
    """Canonical Jacobian field name: `"{field}_J_{var}"`."""

    field = str(field)
    var = str(var)
    if field == "":
        raise ValueError("jac_field: field must be non-empty.")
    if var == "":
        raise ValueError("jac_field: var must be non-empty.")
    return f"{field}_J_{var}"


def is_jac_field(field: str) -> bool:
    base, sep, var = str(field).partition("_J_")
    return bool(sep) and base != "" and var != ""


def split_jac_field(field: str) -> tuple[str, str]:
    base, sep, var = str(field).partition("_J_")
    if not sep or base == "" or var == "":
        raise ValueError(f"split_jac_field: not a jacobian field: {field!r}")
    return base, var


def canonical_field_name(field: str) -> str:
    f = str(field)
    if is_jac_field(f):
        base, var = split_jac_field(f)
        return jac_field(canonical_field_name(base), var=var)
    if _is_deprecated_torque_alias(f):
        raise ValueError(
            "canonical_field_name: deprecated field alias "
            f"{f!r}. Use canonical field names (e.g. 'torque', 'momentum', 'force', 'torque_d1')."
        )
    alias = FIELD_ALIASES.get(f, f)
    return _canonicalize_torque_field(alias)


def canonical_dtype_name(dtype: str) -> str:
    d = str(dtype).strip()
    if d == "":
        raise ValueError("canonical_dtype_name: dtype must be non-empty.")
    if d in _DEPRECATED_DTYPE_ALIASES:
        raise ValueError(
            f"canonical_dtype_name: deprecated dtype alias {d!r}. "
            f"Use canonical dtype name {DTYPE_COORD!r}."
        )
    return d


def torque_derivative_order(field: str) -> int | None:
    field_name = canonical_field_name(str(field))
    if field_name == "torque":
        return 0
    m = _TORQUE_D_PATTERN.fullmatch(field_name)
    if m is None:
        return None
    order = int(m.group(1))
    if order <= 0:
        return None
    return order


def make_key(
    *,
    k: int,
    owner_type: str,
    owner_name: str,
    dtype: str,
    field: str,
    frame: str | None = None,
    rel_frame: str | None = None,
) -> StateKey:
    dtype_name = canonical_dtype_name(str(dtype))
    field_name = canonical_field_name(str(field))
    return StateKey(
        k=int(k),
        owner=OwnerKey(owner_type=str(owner_type), owner_name=str(owner_name)),
        dtype=dtype_name,
        field=field_name,
        frame=frame,
        rel_frame=rel_frame,
    )


def make_jac_key(
    *,
    k: int,
    owner_type: str,
    owner_name: str,
    dtype: str,
    field: str,
    var: str,
    frame: str | None = None,
    rel_frame: str | None = None,
) -> StateKey:
    return make_key(
        k=k,
        owner_type=owner_type,
        owner_name=owner_name,
        dtype=dtype,
        field=jac_field(field, var=var),
        frame=frame,
        rel_frame=rel_frame,
    )


def joint_q_key(*, k: int = 0, owner_name: str = DEFAULT_ROBOT_NAME) -> StateKey:
    return make_key(
        k=int(k),
        owner_type="total_joint",
        owner_name=str(owner_name),
        dtype=DTYPE_COORD,
        field="q",
    )


def joint_q_jac_key(
    *,
    k: int = 0,
    var: str = "q",
    owner_name: str = DEFAULT_ROBOT_NAME,
) -> StateKey:
    return make_jac_key(
        k=int(k),
        owner_type="total_joint",
        owner_name=str(owner_name),
        dtype=DTYPE_COORD,
        field="q",
        var=str(var),
    )


# ---------------------------------------------------------------------
# Canonical vectorization helpers (rotation + pose)
# ---------------------------------------------------------------------
def rot_flat(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=float)
    if r.shape == (3, 3):
        return r.reshape(-1)
    r = r.reshape(-1)
    if r.size != 9:
        raise ValueError(f"rot_flat: expected (3,3) or (9,), got {rot!r}")
    return r


def rot_mat(rot9: np.ndarray) -> np.ndarray:
    r = np.asarray(rot9, dtype=float).reshape(-1)
    if r.size != 9:
        raise ValueError(f"rot_mat: expected (9,), got {rot9!r}")
    return r.reshape(3, 3)


def pack_frame(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    p = np.asarray(pos, dtype=float).reshape(-1)
    if p.size != 3:
        raise ValueError(f"pack_frame: pos must be (3,), got {pos!r}")
    r = rot_flat(rot)
    return np.concatenate([p, r], axis=0)


def unpack_frame(frame12: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(frame12, dtype=float).reshape(-1)
    if x.size != 12:
        raise ValueError(f"unpack_frame: expected (12,), got {frame12!r}")
    pos = x[:3].copy()
    rot = x[3:].reshape(3, 3).copy()
    return pos, rot
