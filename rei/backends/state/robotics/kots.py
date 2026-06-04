from __future__ import annotations

from collections.abc import Mapping, Sequence, Iterable
from dataclasses import dataclass
import re
from typing import Any
import warnings

import numpy as np

from ....core.state_cache import StateKey
from ....core.state_schema import (
    DYNAMICS_FIELDS,
    DTYPE_COORD,
    DTYPE_DYNAMICS,
    DTYPE_KINEMATICS,
    canonical_field_name,
    split_jac_field,
    torque_derivative_field,
    torque_derivative_order,
)
from ....core.trajectory import TrajectoryMap
from ..dispatch.template import BackendDispatchStateBuilder

try:
    from robokots.core.state import StateType
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "`rei.backends.state.robotics.kots` requires the robotics RoboKots bindings. "
        "Install RoboKots (e.g. via github) and retry."
    ) from e

Array = np.ndarray
STATE_JACOBIAN_VAR = "state"
_ROBOKOTS_TORQUE_DIFF_PATTERN = re.compile(r"^torque_diff([1-9][0-9]*)$")
_ROBOKOTS_JACOBIAN_MUL_METHODS = ("jacobian_mul", "matvec", "jacobian_matvec")
_KOTS_JACOBIAN_STRATEGIES = ("dense", "mul")


def _normalize_kots_jacobian_strategy(strategy: str | None, *, prefer_matvec_jacobian: bool) -> str:
    if prefer_matvec_jacobian:
        warnings.warn(
            "KotsTrajectoryStateBuilder: prefer_matvec_jacobian is deprecated; "
            "use jacobian_strategy='mul' or 'dense' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if strategy is None:
        return "mul"
    name = str(strategy).strip().lower()
    if name == "":
        raise ValueError("KotsTrajectoryStateBuilder: jacobian_strategy must be non-empty.")
    if name == "matvec":
        name = "mul"
    if name not in _KOTS_JACOBIAN_STRATEGIES:
        allowed = ", ".join(repr(v) for v in _KOTS_JACOBIAN_STRATEGIES)
        raise ValueError(f"KotsTrajectoryStateBuilder: jacobian_strategy must be one of {allowed}, got {strategy!r}.")
    return name


@dataclass(frozen=True)
class KotsFieldFamily:
    field: str


@dataclass(frozen=True)
class _TotalJointDynamicsStateRef:
    field: str
    refs: tuple[Any, ...]


# kots.py 内で「どの field ファミリを提供するか」を宣言する登録リスト。
KOTS_DEFAULT_FIELD_FAMILIES: tuple[KotsFieldFamily, ...] = (
    KotsFieldFamily(field="pos"),
    KotsFieldFamily(field="rot"),
    KotsFieldFamily(field="frame"),
)


class KotsStateBuilder(BackendDispatchStateBuilder):
    """RoboKots/Kots -> `build_state()` bridge with StateKey-based automatic dispatch."""

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        q_var: str = "q",
        fields: Sequence[str] | None = None,
        dynamics_fields: Sequence[str] | None = DYNAMICS_FIELDS,
        dynamics_owner_type: str = "total_joint",
        prefer_matvec_jacobian: bool = False,
    ) -> None:
        super().__init__(model, data, q_var=q_var)
        self.dtype = DTYPE_KINEMATICS
        self.owner_type = "link"
        self.dynamics_owner_type = str(dynamics_owner_type)
        if self.dynamics_owner_type == "":
            raise ValueError("KotsStateBuilder: dynamics_owner_type must be non-empty.")
        self._needs_dynamics_update = False
        self._model_dof_cache: int | None = None
        self._model_order_cache: int | None = None
        self.prefer_matvec_jacobian = bool(prefer_matvec_jacobian)

        family_map = {spec.field: spec for spec in KOTS_DEFAULT_FIELD_FAMILIES}
        selected_fields = [spec.field for spec in KOTS_DEFAULT_FIELD_FAMILIES] if fields is None else [str(f) for f in fields]
        if len(selected_fields) == 0:
            raise ValueError("KotsStateBuilder: fields must be non-empty.")

        self.field_to_jac: dict[str, str] = {}
        for field in selected_fields:
            spec = family_map.get(field, None)
            if spec is None:
                supported = ", ".join(sorted(family_map.keys()))
                raise ValueError(
                    f"KotsStateBuilder: unsupported field {field!r}. "
                    f"Supported fields: {supported}."
                )
            _value_name, jac_name = self.register_value_and_jac(
                dtype=self.dtype,
                owner_type=self.owner_type,
                field=spec.field,
                value_handler=self._handle_value,
                jac_handler=self._handle_jac,
                jacobian_wrt=STATE_JACOBIAN_VAR,
            )
            self.field_to_jac[spec.field] = jac_name

        if dynamics_fields is not None:
            dyn_fields_raw = [str(f) for f in dynamics_fields]
            dyn_fields = []
            for field in dyn_fields_raw:
                if field == "":
                    raise ValueError("KotsStateBuilder: dynamics field names must be non-empty.")
                dyn_fields.append(canonical_field_name(field))
            dyn_fields = list(dict.fromkeys(dyn_fields))
            if len(dyn_fields) == 0:
                raise ValueError("KotsStateBuilder: dynamics_fields must be non-empty when provided.")
            self._needs_dynamics_update = True
            for field in dyn_fields:
                self.register_value_and_jac(
                    dtype=DTYPE_DYNAMICS,
                    owner_type=self.dynamics_owner_type,
                    field=field,
                    value_handler=self._handle_value,
                    jac_handler=self._handle_jac,
                    jacobian_wrt=STATE_JACOBIAN_VAR,
                )

    def _update_dynamics_if_available(self) -> bool:
        if not self._needs_dynamics_update:
            return False
        for name in ("dynamics", "compute_dynamics", "update_dynamics"):
            fn = getattr(self.model, name, None)
            if not callable(fn):
                continue
            try:
                fn()
            except TypeError:
                continue
            return True
        return False

    def _update_kinematics(self, q: Array) -> None:
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        dof = self._model_dof()
        order = self._model_order()

        if q_vec.size == dof * order:
            motion = q_vec
        elif q_vec.size == dof:
            motion = self._expand_coordinate_motion(q_vec, dof=dof, order=order)
        else:
            raise ValueError(
                "KotsStateBuilder: unexpected q size. "
                f"Expected dof ({dof}) or dof*order ({dof * order}), got {q_vec.size}."
            )

        self.model.import_motions(motion)
        if self._needs_dynamics_update:
            if self._update_dynamics_if_available():
                return

        self.model.kinematics()
        if self._needs_dynamics_update:
            self._update_dynamics_if_available()

    def _model_dof(self) -> int:
        if self._model_dof_cache is not None:
            return int(self._model_dof_cache)
        dof_fn = getattr(self.model, "dof", None)
        if callable(dof_fn):
            dof = int(dof_fn())
            self._model_dof_cache = dof
            return dof
        robot = getattr(self.model, "robot_", None)
        if robot is not None and hasattr(robot, "dof"):
            dof = int(getattr(robot, "dof"))
            self._model_dof_cache = dof
            return dof
        raise ValueError("KotsStateBuilder: unable to resolve model dof.")

    def _model_order(self) -> int:
        if self._model_order_cache is not None:
            return int(self._model_order_cache)
        order_fn = getattr(self.model, "order", None)
        if callable(order_fn):
            order = int(order_fn())
        else:
            order = int(getattr(self.model, "order_", 1))
        if order < 1:
            raise ValueError(f"KotsStateBuilder: model order must be >= 1, got {order}.")
        self._model_order_cache = int(order)
        return order

    def _expand_coordinate_motion(self, q: Array, *, dof: int, order: int) -> Array:
        motion = np.zeros(dof * order, dtype=float)
        robot = getattr(self.model, "robot_", None)
        if robot is None:
            for i in range(min(q.size, dof)):
                motion[i * order] = float(q[i])
            return motion

        owners = [*getattr(robot, "links", []), *getattr(robot, "joints", [])]
        owners = [owner for owner in owners if int(getattr(owner, "dof", 0)) > 0]
        owners.sort(key=lambda owner: int(getattr(owner, "dof_index", 0)))

        cursor = 0
        for owner in owners:
            owner_dof = int(getattr(owner, "dof", 0))
            dof_index = int(getattr(owner, "dof_index", 0))
            start = dof_index * order
            stop = start + owner_dof
            if stop > motion.size:
                raise ValueError("KotsStateBuilder: invalid dof_index/dof in robot structure.")
            motion[start:stop] = q[cursor : cursor + owner_dof]
            cursor += owner_dof

        if cursor != q.size:
            raise ValueError(
                "KotsStateBuilder: failed to map q into motion coordinates. "
                f"Mapped {cursor} elements from q size {q.size}."
            )
        return motion

    def _resolve_state_ref(self, key: StateKey) -> Any:
        owner = getattr(key, "owner", None)
        owner_type = getattr(owner, "owner_type", None)
        owner_name = getattr(owner, "owner_name", None)
        if not isinstance(owner_name, str) or owner_name == "":
            raise ValueError(f"Kots backend expects non-empty owner_name in key, got: {key!r}")

        state_field = self._state_field_name(key.field)
        if owner_type == "total_joint" and getattr(key, "dtype", None) == DTYPE_COORD and key.field == "q":
            # Joint-q terms are computed directly from optimization variables; no backend state query is required.
            return ("total_joint", owner_name, "q")

        if owner_type == self.dynamics_owner_type and getattr(key, "dtype", None) == DTYPE_DYNAMICS:
            # RoboKots does not robustly support world dynamics queries for owner_type="total_joint".
            # Expand to per-joint queries and stack them in dof order.
            joint_refs = self._resolve_total_joint_dynamics_refs(state_field=state_field, key=key)
            if joint_refs is not None:
                return _TotalJointDynamicsStateRef(field=state_field, refs=joint_refs)

        route = self._route_for_key(key)
        if route is None or route not in self._dispatch:
            raise ValueError(f"Kots backend has no handler route for key: {key!r}")

        frame_name = getattr(key, "frame", None) or "world"
        return self._make_state_type(
            owner_type=str(owner_type),
            owner_name=owner_name,
            state_field=state_field,
            frame_name=str(frame_name),
        )

    def _resolve_total_joint_dynamics_refs(self, *, state_field: str, key: StateKey) -> tuple[Any, ...] | None:
        joints = self._dof_sorted_joints()
        if joints is None:
            return None

        if len(joints) == 0:
            return tuple()

        frame_name = self._total_joint_dynamics_frame_name(state_field=state_field, key=key)

        refs: list[Any] = []
        for joint in joints:
            joint_name = str(getattr(joint, "name", ""))
            if joint_name == "":
                raise ValueError("KotsStateBuilder: joint.name must be non-empty for dynamics expansion.")
            refs.append(
                self._make_state_type(
                    owner_type="joint",
                    owner_name=joint_name,
                    state_field=state_field,
                    frame_name=frame_name,
                )
            )
        return tuple(refs)

    @staticmethod
    def _state_field_name(field: Any) -> str:
        return canonical_field_name(str(field))

    @staticmethod
    def _fallback_backend_field_name(state_field: str) -> str:
        deriv_order = torque_derivative_order(state_field)
        if isinstance(deriv_order, int) and deriv_order > 0:
            return f"torque_diff{deriv_order}"
        return state_field

    def _make_state_type(
        self,
        *,
        owner_type: str,
        owner_name: str,
        state_field: str,
        frame_name: str | None,
    ) -> Any:
        try:
            return StateType(owner_type, owner_name, state_field, frame_name)
        except KeyError:
            fallback_field = self._fallback_backend_field_name(state_field)
            if fallback_field == state_field:
                raise
            return StateType(owner_type, owner_name, fallback_field, frame_name)

    @staticmethod
    def _torque_derivative_order_from_state_data_type(data_type: str) -> int | None:
        try:
            deriv_order = torque_derivative_order(data_type)
        except ValueError:
            deriv_order = None
        if isinstance(deriv_order, int):
            return deriv_order
        m = _ROBOKOTS_TORQUE_DIFF_PATTERN.fullmatch(data_type)
        if m is None:
            return None
        return int(m.group(1))

    def _dof_sorted_joints(self) -> list[Any] | None:
        robot = getattr(self.model, "robot_", None)
        if robot is None:
            return None
        joints_raw = getattr(robot, "joints", None)
        if joints_raw is None:
            return None
        joints = [joint for joint in joints_raw if int(getattr(joint, "dof", 0)) > 0]
        joints.sort(key=lambda joint: int(getattr(joint, "dof_index", 0)))
        return joints

    def _total_joint_dynamics_frame_name(self, *, state_field: str, key: StateKey) -> str | None:
        frame_name = getattr(key, "frame", None) or "world"
        # torque family does not need world-frame conversion in RoboKots get_value path.
        # Keeping frame unset avoids a known owner_type='total_joint' world-path bug upstream.
        if "torque" in state_field:
            return None
        return str(frame_name)

    @staticmethod
    def _state_ref_data_type(state_ref: Any) -> str | None:
        for attr in ("data_type", "field", "field_", "dtype"):
            value = getattr(state_ref, attr, None)
            if isinstance(value, str) and value != "":
                return value
        return None

    @staticmethod
    def _state_ref_owner_type(state_ref: Any) -> str | None:
        value = getattr(state_ref, "owner_type", None)
        if isinstance(value, str) and value != "":
            return value
        return None

    @staticmethod
    def _state_ref_owner_name(state_ref: Any) -> str | None:
        value = getattr(state_ref, "owner_name", None)
        if isinstance(value, str) and value != "":
            return value
        return None

    @staticmethod
    def _state_ref_frame_name(state_ref: Any) -> str | None:
        for attr in ("frame", "frame_name"):
            value = getattr(state_ref, attr, None)
            if isinstance(value, str) and value != "":
                return value
        return None

    def _raise_missing_state_key(self, *, state_ref: Any, cause: KeyError) -> None:
        data_type = self._state_ref_data_type(state_ref)
        deriv_order = None if data_type is None else self._torque_derivative_order_from_state_data_type(data_type)
        if isinstance(deriv_order, int) and deriv_order > 0:
            field = torque_derivative_field(deriv_order)
            required_model_order = deriv_order + 3
            raise ValueError(
                "KotsStateBuilder: dynamics field "
                f"{field!r} requires RoboKots model order >= {required_model_order}. "
                f"Current model order is {self._model_order()}."
            ) from cause
        raise cause

    def _value_from_single_state_ref(self, state_ref: Any) -> Array:
        try:
            return np.asarray(self.model.state_info(state_ref), dtype=float).reshape(-1)
        except KeyError as e:
            self._raise_missing_state_key(state_ref=state_ref, cause=e)

    def _value_from_state_ref(self, state_ref: Any) -> Array:
        total_joint_ref = self._as_total_joint_dynamics_state_ref(state_ref)
        if total_joint_ref is not None:
            return self._concat_total_joint_values(total_joint_ref.refs)

        return self._value_from_single_state_ref(state_ref)

    def _value_from_state_refs(self, refs: tuple[Any, ...]) -> Array:
        if len(refs) == 0:
            return np.zeros((0,), dtype=float)

        fn = getattr(self.model, "state_info_list", None)
        if callable(fn):
            try:
                return np.asarray(fn(list(refs)), dtype=float).reshape(-1)
            except (KeyError, ValueError, TypeError, RuntimeError):
                pass

        return self._concat_total_joint_values(refs)

    def _jac_from_single_state_ref(self, state_ref: Any) -> Array:
        try:
            J = np.asarray(self.model.jacobian(state_ref), dtype=float)
        except KeyError as e:
            self._raise_missing_state_key(state_ref=state_ref, cause=e)
        return self._normalize_jacobian_shape(J, state_ref=state_ref)

    def _normalize_jacobian_shape(self, J_raw: Any, *, state_ref: Any) -> Array:
        J = np.asarray(J_raw, dtype=float)
        if J.ndim != 2:
            raise ValueError(f"Kots Jacobian must be 2D, got shape {J.shape}.")

        # Fast path: if one axis matches motion-space dimension, avoid an extra
        # state_info() call just for orientation inference.
        n_motion = int(self._model_dof() * self._model_order())
        if J.shape[1] == n_motion:
            return J
        if J.shape[0] == n_motion:
            return J.T

        total_joint_ref = self._as_total_joint_dynamics_state_ref(state_ref)
        if total_joint_ref is None:
            m = int(self._value_from_single_state_ref(state_ref).size)
        else:
            m = int(self._value_from_state_refs(total_joint_ref.refs).size)
        if J.shape[0] == m:
            return J
        if J.shape[1] == m:
            return J.T
        raise ValueError(f"Kots Jacobian must be ({m},n) or (n,{m}), got {J.shape}.")

    def _jac_from_state_refs(self, refs: tuple[Any, ...]) -> Array:
        if len(refs) == 0:
            return np.zeros((0, self._model_dof() * self._model_order()), dtype=float)

        try:
            J = np.asarray(self.model.jacobian(list(refs)), dtype=float)
            return self._normalize_jacobian_shape(J, state_ref=_TotalJointDynamicsStateRef(field="", refs=refs))
        except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
            return self._stack_total_joint_jacobians_fallback(refs)

    def _matvec_from_single_state_ref(self, state_ref: Any, vec: Array) -> Array:
        v = np.asarray(vec, dtype=float).reshape(-1)
        errors: list[Exception] = []
        for name in _ROBOKOTS_JACOBIAN_MUL_METHODS:
            fn = getattr(self.model, name, None)
            if not callable(fn):
                continue
            for args in ((state_ref, v), (v, state_ref)):
                try:
                    out = np.asarray(fn(*args), dtype=float).reshape(-1)
                except (KeyError, ValueError, TypeError, RuntimeError) as e:
                    errors.append(e)
                    continue
                return out
        if len(errors) > 0:
            raise errors[-1]
        raise AttributeError("KotsStateBuilder: model does not expose jacobian_mul(state_ref, vec).")

    def _jac_from_matrix_mul_single_state_ref(self, state_ref: Any, cols: Array) -> Array:
        C = np.asarray(cols, dtype=float)
        if C.ndim != 2:
            raise ValueError(f"Kots jacobian_mul columns must be 2D, got shape {C.shape}.")

        errors: list[Exception] = []
        fn = getattr(self.model, "jacobian_mul", None)
        if not callable(fn):
            raise AttributeError("KotsStateBuilder: model does not expose jacobian_mul(state_ref, cols).")

        for args in ((state_ref, C), (C, state_ref)):
            try:
                out = np.asarray(fn(*args), dtype=float)
            except (KeyError, ValueError, TypeError, RuntimeError) as e:
                errors.append(e)
                continue

            if out.ndim == 1 and C.shape[1] == 1:
                return out.reshape(-1, 1)
            if out.ndim != 2:
                errors.append(ValueError(f"Kots jacobian_mul output must be 2D, got shape {out.shape}."))
                continue
            if out.shape[1] == C.shape[1]:
                return out
            if out.shape[0] == C.shape[1]:
                return out.T
            errors.append(
                ValueError(
                    "Kots jacobian_mul output column mismatch. "
                    f"Expected {C.shape[1]} columns, got shape {out.shape}."
                )
            )

        if len(errors) > 0:
            raise errors[-1]
        raise AttributeError("KotsStateBuilder: model does not expose jacobian_mul(state_ref, cols).")

    def _jac_from_matvec_single_state_ref(self, state_ref: Any, cols: Array) -> Array:
        C = np.asarray(cols, dtype=float)
        if C.ndim != 2:
            raise ValueError(f"Kots matvec columns must be 2D, got shape {C.shape}.")
        try:
            return self._jac_from_matrix_mul_single_state_ref(state_ref, C)
        except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
            pass

        parts: list[Array] = []
        for j in range(int(C.shape[1])):
            try:
                y = self._matvec_from_single_state_ref(state_ref, C[:, j])
            except KeyError as e:
                self._raise_missing_state_key(state_ref=state_ref, cause=e)
            parts.append(np.asarray(y, dtype=float).reshape(-1))
        if len(parts) == 0:
            m = int(self._value_from_single_state_ref(state_ref).size)
            return np.zeros((m, 0), dtype=float)
        m = int(parts[0].size)
        for part in parts[1:]:
            if int(part.size) != m:
                raise ValueError(
                    "KotsStateBuilder: inconsistent matvec output size while assembling Jacobian. "
                    f"Expected {m}, got {part.size}."
                )
        return np.column_stack(parts)

    def _handle_value(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del q, key
        return self._value_from_state_ref(state_ref)

    def _handle_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del q
        total_joint_ref = self._as_total_joint_dynamics_state_ref(state_ref)
        if total_joint_ref is not None:
            return self._stack_total_joint_jacobians(total_joint_ref.refs)

        J = self._jac_from_single_state_ref(state_ref)
        return self._rotate_link_kinematics_jacobian_to_world(J=J, key=key, state_ref=state_ref)

    def _link_world_rotation(self, *, state_ref: Any) -> Array | None:
        owner_type = self._state_ref_owner_type(state_ref)
        if owner_type != self.owner_type:
            return None

        owner_name = self._state_ref_owner_name(state_ref)
        if not isinstance(owner_name, str) or owner_name == "":
            return None

        frame_name = self._state_ref_frame_name(state_ref) or "world"
        try:
            rot_ref = self._make_state_type(
                owner_type=owner_type,
                owner_name=owner_name,
                state_field="rot",
                frame_name=frame_name,
            )
            rot_flat = self._value_from_single_state_ref(rot_ref)
        except Exception:
            return None

        if int(rot_flat.size) != 9:
            return None
        return np.asarray(rot_flat, dtype=float).reshape(3, 3)

    def _rotate_link_kinematics_jacobian_to_world(
        self,
        *,
        J: Array,
        key: StateKey,
        state_ref: Any,
    ) -> Array:
        if getattr(key, "dtype", None) != DTYPE_KINEMATICS:
            return J
        owner = getattr(key, "owner", None)
        if getattr(owner, "owner_type", None) != self.owner_type:
            return J

        rot = self._link_world_rotation(state_ref=state_ref)
        if rot is None:
            return J

        rows = int(J.shape[0])
        if rows % 3 != 0:
            return J

        # RoboKots reports link Jacobians in the link-local frame.
        # Convert each 3D block into world-frame coordinates for backend parity.
        J_world = np.asarray(J, dtype=float).copy()
        for i in range(0, rows, 3):
            J_world[i : i + 3, :] = rot @ J_world[i : i + 3, :]
        return J_world

    def _rotate_link_kinematics_rhs_to_local(
        self,
        *,
        rhs: Array,
        key: StateKey,
        state_ref: Any,
    ) -> Array:
        if getattr(key, "dtype", None) != DTYPE_KINEMATICS:
            return rhs
        owner = getattr(key, "owner", None)
        if getattr(owner, "owner_type", None) != self.owner_type:
            return rhs

        rot = self._link_world_rotation(state_ref=state_ref)
        if rot is None:
            return rhs

        R = np.asarray(rhs, dtype=float)
        rows = int(R.shape[0])
        if rows % 3 != 0:
            return R

        R_local = R.copy()
        for i in range(0, rows, 3):
            R_local[i : i + 3, ...] = rot.T @ R_local[i : i + 3, ...]
        return R_local

    def _transpose_matvec_from_single_state_ref(self, state_ref: Any, rhs: Array) -> Array:
        R = np.asarray(rhs, dtype=float)
        if R.ndim not in (1, 2):
            raise ValueError(f"Kots jacobian_transpose_mul rhs must be 1D or 2D, got shape {R.shape}.")

        errors: list[Exception] = []
        fn = getattr(self.model, "jacobian_transpose_mul", None)
        if not callable(fn):
            raise AttributeError("KotsStateBuilder: model does not expose jacobian_transpose_mul(state_ref, rhs).")

        for args in ((state_ref, R), (R, state_ref)):
            try:
                out = np.asarray(fn(*args), dtype=float)
            except (KeyError, ValueError, TypeError, RuntimeError) as e:
                errors.append(e)
                continue
            if out.ndim not in (1, 2):
                errors.append(
                    ValueError(f"Kots jacobian_transpose_mul output must be 1D or 2D, got shape {out.shape}.")
                )
                continue
            return out

        if len(errors) > 0:
            raise errors[-1]
        raise AttributeError("KotsStateBuilder: model does not expose jacobian_transpose_mul(state_ref, rhs).")

    def _transpose_matvec_from_state_ref(self, state_ref: Any, rhs: Array) -> Array:
        total_joint_ref = self._as_total_joint_dynamics_state_ref(state_ref)
        R = np.asarray(rhs, dtype=float)
        if R.ndim not in (1, 2):
            raise ValueError(f"Kots jacobian_transpose_mul rhs must be 1D or 2D, got shape {R.shape}.")

        if total_joint_ref is None:
            return self._transpose_matvec_from_single_state_ref(state_ref, R)

        try:
            return np.asarray(self._transpose_matvec_from_single_state_ref(list(total_joint_ref.refs), R), dtype=float)
        except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
            pass

        parts: list[Array] = []
        offset = 0
        for ref in total_joint_ref.refs:
            m = int(self._value_from_single_state_ref(ref).size)
            stop = offset + m
            if stop > int(R.shape[0]):
                raise ValueError(
                    "KotsStateBuilder: rhs is too short for total_joint transpose multiply. "
                    f"Need at least {stop} rows, got {R.shape[0]}."
                )
            parts.append(
                np.asarray(
                    self._transpose_matvec_from_single_state_ref(ref, R[offset:stop, ...]),
                    dtype=float,
                )
            )
            offset = stop

        if offset != int(R.shape[0]):
            raise ValueError(
                "KotsStateBuilder: rhs size mismatch for total_joint transpose multiply. "
                f"Consumed {offset} rows, got {R.shape[0]}."
            )
        if len(parts) == 0:
            return np.zeros((self._model_dof() * self._model_order(),), dtype=float)
        return np.sum(np.stack(parts, axis=0), axis=0)

    def jacobian_transpose_mul(
        self,
        q: Array,
        key: StateKey,
        rhs: Array,
        *,
        update_kinematics: bool = True,
    ) -> Array:
        """Compute `J(key).T @ rhs` with RoboKots' transpose multiply when available."""

        if not self._accept_required_key(key):
            raise ValueError(f"KotsStateBuilder: invalid state key for jacobian_transpose_mul: {key!r}")
        route = self._route_for_key(key)
        entry = None if route is None else self._dispatch.get(route, None)
        if entry is None:
            raise ValueError(f"KotsStateBuilder: no value handler route for key: {key!r}")

        q_vec = self._extract_q(np.asarray(q, dtype=float).reshape(-1), pack=None)
        if update_kinematics:
            self._update_kinematics(q_vec)
        state_ref = self._state_ref(key, state_ref_field=entry.state_ref_field)
        rhs_local = self._rotate_link_kinematics_rhs_to_local(
            rhs=np.asarray(rhs, dtype=float),
            key=key,
            state_ref=state_ref,
        )
        return self._transpose_matvec_from_state_ref(state_ref, rhs_local)

    @staticmethod
    def _as_total_joint_dynamics_state_ref(state_ref: Any) -> _TotalJointDynamicsStateRef | None:
        if isinstance(state_ref, _TotalJointDynamicsStateRef):
            return state_ref
        if (
            isinstance(state_ref, tuple)
            and len(state_ref) == 3
            and state_ref[0] == "total_joint_dynamics"
        ):
            return _TotalJointDynamicsStateRef(field=str(state_ref[1]), refs=tuple(state_ref[2]))
        return None

    def _concat_total_joint_values(self, refs: tuple[Any, ...]) -> Array:
        parts = [self._value_from_single_state_ref(ref) for ref in refs]
        if len(parts) == 0:
            return np.zeros((0,), dtype=float)
        return np.concatenate(parts, axis=0)

    def _stack_total_joint_jacobians(self, refs: tuple[Any, ...]) -> Array:
        return self._jac_from_state_refs(refs)

    def _stack_total_joint_jacobians_fallback(self, refs: tuple[Any, ...]) -> Array:
        blocks = [self._jac_from_single_state_ref(ref) for ref in refs]
        if len(blocks) == 0:
            return np.zeros((0, self._model_dof() * self._model_order()), dtype=float)
        ncols = int(blocks[0].shape[1])
        for block in blocks[1:]:
            if int(block.shape[1]) != ncols:
                raise ValueError(
                    "KotsStateBuilder: inconsistent Jacobian column size while stacking total_joint dynamics. "
                    f"Expected {ncols}, got {block.shape[1]}."
                )
        return np.vstack(blocks)


class KotsTrajectoryStateBuilder(KotsStateBuilder):
    """RoboKots trajectory builder with trajectory parameterization.

    Decision variable is `p` (configurable by `p_var`), and generalized coordinates are:

      q(k) = trajectory_map.q_at(p, k)
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        trajectory_map: TrajectoryMap,
        trajectory_derivative_maps: Mapping[int, TrajectoryMap] | None = None,
        p_var: str = "p",
        fields: Sequence[str] | None = None,
        dynamics_fields: Sequence[str] | None = DYNAMICS_FIELDS,
        dynamics_owner_type: str = "total_joint",
        prefer_matvec_jacobian: bool = False,
        jacobian_strategy: str | None = None,
    ) -> None:
        self.trajectory_map = trajectory_map
        self.p_var = str(p_var)
        if self.p_var == "":
            raise ValueError("KotsTrajectoryStateBuilder: p_var must be non-empty.")
        self.jacobian_strategy = _normalize_kots_jacobian_strategy(
            jacobian_strategy,
            prefer_matvec_jacobian=prefer_matvec_jacobian,
        )
        self.trajectory_derivative_maps: dict[int, TrajectoryMap] = {0: trajectory_map}
        if trajectory_derivative_maps is not None:
            for order_raw, traj in trajectory_derivative_maps.items():
                order = int(order_raw)
                if order < 0:
                    raise ValueError(f"KotsTrajectoryStateBuilder: derivative order must be >= 0, got {order}.")
                self.trajectory_derivative_maps[order] = traj
        self._validate_derivative_maps()

        super().__init__(
            model,
            data,
            q_var=self.p_var,
            fields=fields,
            dynamics_fields=dynamics_fields,
            dynamics_owner_type=dynamics_owner_type,
            prefer_matvec_jacobian=prefer_matvec_jacobian,
        )
        self.register_value_and_jac(
            dtype=DTYPE_COORD,
            owner_type="total_joint",
            field="q",
            value_handler=self._handle_joint_q_value,
            jac_handler=self._handle_joint_q_jac,
            jacobian_wrt=STATE_JACOBIAN_VAR,
        )

    def _validate_derivative_maps(self) -> None:
        base = self.trajectory_derivative_maps.get(0, None)
        if base is None:
            raise ValueError("KotsTrajectoryStateBuilder: derivative map for order 0 is required.")
        for order, traj in self.trajectory_derivative_maps.items():
            if traj.p_dim != base.p_dim:
                raise ValueError(
                    "KotsTrajectoryStateBuilder: derivative map p_dim mismatch. "
                    f"order={order}, expected {base.p_dim}, got {traj.p_dim}."
                )
            if traj.steps != base.steps or traj.q_dim != base.q_dim:
                raise ValueError(
                    "KotsTrajectoryStateBuilder: derivative map shape mismatch. "
                    f"order={order}, expected steps={base.steps}, q_dim={base.q_dim}, "
                    f"got steps={traj.steps}, q_dim={traj.q_dim}."
                )

    def _handle_joint_q_value(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        return np.asarray(q, dtype=float).reshape(-1).copy()

    def _handle_joint_q_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key, state_ref
        n = int(np.asarray(q, dtype=float).reshape(-1).size)
        return np.eye(n, dtype=float)

    def _compose_motion_and_jac(self, p: Array, *, k: int) -> tuple[Array, Array]:
        p_vec = np.asarray(p, dtype=float).reshape(-1)
        dof = int(self.trajectory_map.q_dim)
        order = self._model_order()
        motion = np.zeros((dof * order,), dtype=float)
        dmotion_dp = np.zeros((dof * order, int(self.trajectory_map.p_dim)), dtype=float)

        for deriv_order, traj in self.trajectory_derivative_maps.items():
            if deriv_order >= order:
                continue
            q_r = np.asarray(traj.q_at(p_vec, k), dtype=float).reshape(-1)
            J_r = np.asarray(traj.dqdp_at(k), dtype=float)
            if q_r.size != dof:
                raise ValueError(
                    "KotsTrajectoryStateBuilder: derivative map q size mismatch. "
                    f"order={deriv_order}, expected {dof}, got {q_r.size}."
                )
            if J_r.shape != (dof, self.trajectory_map.p_dim):
                raise ValueError(
                    "KotsTrajectoryStateBuilder: derivative map jacobian shape mismatch. "
                    f"order={deriv_order}, expected {(dof, self.trajectory_map.p_dim)}, got {J_r.shape}."
                )
            motion[deriv_order::order] = q_r
            dmotion_dp[deriv_order::order, :] = J_r

        return motion, dmotion_dp

    def _chain_param_jac(
        self,
        J_raw: Array,
        *,
        key: StateKey,
        jacobian_wrt: str | None,
        dqdp_k: Array,
        dmotiondp_k: Array,
    ) -> Array:
        Jm = np.asarray(J_raw, dtype=float)
        if Jm.ndim != 2:
            raise ValueError(
                f"KotsTrajectoryStateBuilder: Jacobian must be 2D, got shape {Jm.shape} for key {key!r}."
            )

        wrt = None if jacobian_wrt is None else str(jacobian_wrt)
        if wrt == self.q_var:
            return Jm

        if wrt == STATE_JACOBIAN_VAR:
            if Jm.shape[1] == dqdp_k.shape[0]:
                return Jm @ dqdp_k
            if Jm.shape[1] == dmotiondp_k.shape[0]:
                return Jm @ dmotiondp_k
            # Some RoboKots dynamics Jacobians only depend on lower motion orders
            # even when model.order is higher (e.g. torque at order=4 still uses q,dq,ddq).
            dof = self._model_dof()
            order = self._model_order()
            cols = int(Jm.shape[1])
            if dof > 0 and cols % dof == 0 and dmotiondp_k.shape[0] == dof * order:
                used_order = int(cols // dof)
                if 1 <= used_order <= order:
                    dmotion_reduced = np.zeros((cols, dmotiondp_k.shape[1]), dtype=float)
                    for i in range(dof):
                        src0 = i * order
                        src1 = src0 + used_order
                        dst0 = i * used_order
                        dst1 = dst0 + used_order
                        dmotion_reduced[dst0:dst1, :] = dmotiondp_k[src0:src1, :]
                    return Jm @ dmotion_reduced
            raise ValueError(
                "KotsTrajectoryStateBuilder: Jacobian chain mismatch. "
                f"J_raw has shape {Jm.shape}, dqdp has shape {dqdp_k.shape}, dmotiondp has shape {dmotiondp_k.shape}."
            )

        raise ValueError(
            "KotsTrajectoryStateBuilder: unsupported jacobian_wrt metadata for parameter chain. "
            f"Expected {self.q_var!r} or {STATE_JACOBIAN_VAR!r}, got {wrt!r}."
        )

    def _motion_jacobian_used_order_candidate(
        self,
        *,
        used_order: int,
        dmotiondp_k: Array,
    ) -> Array | None:
        dof = self._model_dof()
        order = self._model_order()
        used_order = int(used_order)
        if dof <= 0 or used_order < 1 or used_order > order:
            return None
        if dmotiondp_k.shape[0] != dof * order:
            return None

        cols = int(dof * used_order)
        dmotion_reduced = np.zeros((cols, dmotiondp_k.shape[1]), dtype=float)
        for i in range(dof):
            src0 = i * order
            src1 = src0 + used_order
            dst0 = i * used_order
            dst1 = dst0 + used_order
            dmotion_reduced[dst0:dst1, :] = dmotiondp_k[src0:src1, :]
        return dmotion_reduced

    def _preferred_motion_jacobian_used_order(self, key: StateKey | None) -> int | None:
        if key is None or getattr(key, "dtype", None) != DTYPE_DYNAMICS:
            return None
        try:
            field, _var = split_jac_field(str(getattr(key, "field", "")))
        except ValueError:
            field = str(getattr(key, "field", ""))
        field = canonical_field_name(field)
        if field == "torque":
            return min(self._model_order(), 3)
        try:
            return min(self._model_order(), torque_derivative_order(field) + 3)
        except ValueError:
            return None

    def _motion_jacobian_chain_candidates(
        self,
        *,
        dqdp_k: Array,
        dmotiondp_k: Array,
        key: StateKey | None = None,
    ) -> tuple[Array, ...]:
        candidates: list[Array] = []
        preferred_order = self._preferred_motion_jacobian_used_order(key)
        if preferred_order is not None:
            preferred = self._motion_jacobian_used_order_candidate(
                used_order=preferred_order,
                dmotiondp_k=np.asarray(dmotiondp_k, dtype=float),
            )
            if preferred is not None:
                candidates.append(preferred)

        candidates.extend([np.asarray(dmotiondp_k, dtype=float), np.asarray(dqdp_k, dtype=float)])
        dof = self._model_dof()
        order = self._model_order()
        if dof > 0 and dmotiondp_k.shape[0] == dof * order:
            for used_order in range(order, 0, -1):
                dmotion_reduced = self._motion_jacobian_used_order_candidate(
                    used_order=used_order,
                    dmotiondp_k=np.asarray(dmotiondp_k, dtype=float),
                )
                if dmotion_reduced is not None:
                    candidates.append(dmotion_reduced)

        unique: list[Array] = []
        seen_shapes: set[tuple[int, int]] = set()
        for cand in candidates:
            shape = (int(cand.shape[0]), int(cand.shape[1]))
            if shape in seen_shapes:
                continue
            unique.append(cand)
            seen_shapes.add(shape)
        return tuple(unique)

    def _param_jac_from_matvec(
        self,
        *,
        key: StateKey,
        state_ref: Any,
        jacobian_wrt: str | None,
        dqdp_k: Array,
        dmotiondp_k: Array,
    ) -> Array:
        wrt = None if jacobian_wrt is None else str(jacobian_wrt)
        if wrt == self.q_var:
            return self._handle_jac(np.zeros((dqdp_k.shape[0],), dtype=float), key, state_ref)
        if wrt != STATE_JACOBIAN_VAR:
            raise ValueError(
                "KotsTrajectoryStateBuilder: unsupported jacobian_wrt metadata for parameter chain. "
                f"Expected {self.q_var!r} or {STATE_JACOBIAN_VAR!r}, got {wrt!r}."
            )

        total_joint_ref = self._as_total_joint_dynamics_state_ref(state_ref)
        last_error: Exception | None = None
        for cols in self._motion_jacobian_chain_candidates(dqdp_k=dqdp_k, dmotiondp_k=dmotiondp_k, key=key):
            try:
                if total_joint_ref is None:
                    Jp = self._jac_from_matvec_single_state_ref(state_ref, cols)
                    return self._rotate_link_kinematics_jacobian_to_world(J=Jp, key=key, state_ref=state_ref)

                try:
                    return self._jac_from_matvec_single_state_ref(list(total_joint_ref.refs), cols)
                except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
                    pass

                blocks = [self._jac_from_matvec_single_state_ref(ref, cols) for ref in total_joint_ref.refs]
                if len(blocks) == 0:
                    return np.zeros((0, cols.shape[1]), dtype=float)
                ncols = int(blocks[0].shape[1])
                for block in blocks[1:]:
                    if int(block.shape[1]) != ncols:
                        raise ValueError(
                            "KotsTrajectoryStateBuilder: inconsistent matvec column size while stacking "
                            f"total_joint dynamics. Expected {ncols}, got {block.shape[1]}."
                        )
                return np.vstack(blocks)
            except (AttributeError, KeyError, ValueError, TypeError, RuntimeError) as e:
                last_error = e
                continue

        if last_error is not None:
            raise last_error
        raise AttributeError("KotsTrajectoryStateBuilder: model does not expose a usable matvec method.")

    def _should_use_param_jacobian_mul(
        self,
        *,
        key: StateKey,
        state_ref: Any,
        dqdp_k: Array,
        dmotiondp_k: Array,
    ) -> bool:
        del key, state_ref, dqdp_k, dmotiondp_k
        return str(getattr(self, "jacobian_strategy", "mul")) != "dense"

    def _param_jac_transpose_mul_from_state_ref(
        self,
        *,
        key: StateKey,
        state_ref: Any,
        rhs: Array,
        dqdp_k: Array,
        dmotiondp_k: Array,
    ) -> Array:
        rhs_local = self._rotate_link_kinematics_rhs_to_local(
            rhs=np.asarray(rhs, dtype=float),
            key=key,
            state_ref=state_ref,
        )
        motion_grad = np.asarray(self._transpose_matvec_from_state_ref(state_ref, rhs_local), dtype=float)

        if motion_grad.ndim == 1:
            for cols in self._motion_jacobian_chain_candidates(dqdp_k=dqdp_k, dmotiondp_k=dmotiondp_k, key=key):
                if int(cols.shape[0]) == int(motion_grad.size):
                    return np.asarray(cols.T @ motion_grad.reshape(-1), dtype=float).reshape(-1)
        elif motion_grad.ndim == 2:
            for cols in self._motion_jacobian_chain_candidates(dqdp_k=dqdp_k, dmotiondp_k=dmotiondp_k, key=key):
                if int(cols.shape[0]) == int(motion_grad.shape[0]):
                    return np.asarray(cols.T @ motion_grad, dtype=float)
        else:
            raise ValueError(
                "KotsTrajectoryStateBuilder: jacobian_transpose_mul output must be 1D or 2D, "
                f"got shape {motion_grad.shape}."
            )

        raise ValueError(
            "KotsTrajectoryStateBuilder: transpose Jacobian chain mismatch. "
            f"transpose output has shape {motion_grad.shape}, dqdp has shape {dqdp_k.shape}, "
            f"dmotiondp has shape {dmotiondp_k.shape}."
        )

    def param_jacobian_transpose_mul(
        self,
        x_all: Array,
        key: StateKey,
        rhs: Array,
        *,
        pack: Any = None,
        time: Any = None,
        update_kinematics: bool = True,
    ) -> Array:
        """Compute `J_p(key).T @ rhs` for a trajectory state key without forming `J_p`."""

        steps = self._expected_steps(time=time)
        if not self._accept_required_key_for_traj(key, steps=steps):
            raise ValueError(
                "KotsTrajectoryStateBuilder: invalid state key for param_jacobian_transpose_mul: "
                f"{key!r}"
            )
        route = self._route_for_key(key)
        entry = None if route is None else self._dispatch.get(route, None)
        if entry is None:
            raise ValueError(f"KotsTrajectoryStateBuilder: no value handler route for key: {key!r}")

        p = self._extract_q(np.asarray(x_all, dtype=float).reshape(-1), pack=pack)
        if p.size != self.trajectory_map.p_dim:
            raise ValueError(
                "KotsTrajectoryStateBuilder: parameter size mismatch. "
                f"Expected p_dim={self.trajectory_map.p_dim}, got {p.size}."
            )

        k = int(key.k)
        dqdp_k = self.trajectory_map.dqdp_at(k)
        motion_k, dmotiondp_k = self._compose_motion_and_jac(p, k=k)
        if update_kinematics:
            self._update_kinematics(motion_k)
        state_ref = self._state_ref(key, state_ref_field=entry.state_ref_field)
        return self._param_jac_transpose_mul_from_state_ref(
            key=key,
            state_ref=state_ref,
            rhs=rhs,
            dqdp_k=dqdp_k,
            dmotiondp_k=dmotiondp_k,
        )

    def _expected_steps(self, *, time: Any = None) -> int:
        steps = int(self.trajectory_map.steps)
        if time is None or not hasattr(time, "N"):
            return steps
        try:
            time_steps = int(time.N) + 1
        except Exception:
            return steps
        if time_steps != steps:
            raise ValueError(
                "KotsTrajectoryStateBuilder: time grid mismatch. "
                f"trajectory_map.steps={steps}, time steps={time_steps} (N+1)."
            )
        return steps

    def _accept_required_key_for_traj(self, key: StateKey, *, steps: int) -> bool:
        if not isinstance(key, StateKey):
            return False
        k = int(getattr(key, "k", -1))
        if k < 0 or k >= steps:
            return False
        dtype = getattr(key, "dtype", None)
        if not isinstance(dtype, str) or dtype == "":
            return False
        owner = getattr(key, "owner", None)
        owner_type = getattr(owner, "owner_type", None)
        owner_name = getattr(owner, "owner_name", None)
        if not isinstance(owner_type, str) or owner_type == "":
            return False
        if not isinstance(owner_name, str) or owner_name == "":
            return False
        field = getattr(key, "field", None)
        if not isinstance(field, str) or field == "":
            return False
        return True

    def _is_param_jac_key(self, key: StateKey) -> bool:
        field = getattr(key, "field", None)
        if not isinstance(field, str) or field == "":
            return False
        try:
            _base, var = split_jac_field(field)
        except ValueError:
            return False
        return var == self.q_var

    def build_state(
        self,
        x_all: Array,
        *,
        pack: Any = None,
        time: Any = None,
        required: Iterable[StateKey] | None = None,
    ) -> dict[StateKey, Any]:
        if required is None:
            return {}

        steps = self._expected_steps(time=time)
        p = self._extract_q(x_all, pack=pack)
        if p.size != self.trajectory_map.p_dim:
            raise ValueError(
                "KotsTrajectoryStateBuilder: parameter size mismatch. "
                f"Expected p_dim={self.trajectory_map.p_dim}, got {p.size}."
            )

        grouped: dict[int, list[tuple[StateKey, Any]]] = {}
        for key in required:
            if not self._accept_required_key_for_traj(key, steps=steps):
                continue
            route = self._route_for_key(key)
            if route is None:
                continue
            entry = self._dispatch.get(route, None)
            if entry is None:
                continue
            grouped.setdefault(int(key.k), []).append((key, entry))

        out: dict[StateKey, Any] = {}
        for k in sorted(grouped.keys()):
            q_k = self.trajectory_map.q_at(p, k)
            dqdp_k = self.trajectory_map.dqdp_at(k)
            motion_k, dmotiondp_k = self._compose_motion_and_jac(p, k=k)
            self._update_kinematics(motion_k)

            for key, entry in grouped[k]:
                state_ref = self._state_ref(key, state_ref_field=entry.state_ref_field)
                is_param_jac = self._is_param_jac_key(key)
                value_from_matvec = False
                if is_param_jac and self._should_use_param_jacobian_mul(
                    key=key,
                    state_ref=state_ref,
                    dqdp_k=dqdp_k,
                    dmotiondp_k=dmotiondp_k,
                ):
                    try:
                        value = self._param_jac_from_matvec(
                            key=key,
                            state_ref=state_ref,
                            jacobian_wrt=entry.jacobian_wrt,
                            dqdp_k=dqdp_k,
                            dmotiondp_k=dmotiondp_k,
                        )
                        value_from_matvec = True
                    except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
                        value = entry.handler(q_k, key, state_ref)
                else:
                    value = entry.handler(q_k, key, state_ref)

                if is_param_jac and not value_from_matvec:
                    value = self._chain_param_jac(
                        value,
                        key=key,
                        jacobian_wrt=entry.jacobian_wrt,
                        dqdp_k=dqdp_k,
                        dmotiondp_k=dmotiondp_k,
                    )

                out[key] = value

        return out


__all__ = [
    "StateType",
    "KotsFieldFamily",
    "KOTS_DEFAULT_FIELD_FAMILIES",
    "KotsStateBuilder",
    "KotsTrajectoryStateBuilder",
]
