from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ....core.state_cache import StateKey
from ....core.state_schema import (
    DTYPE_DYNAMICS,
    DTYPE_KINEMATICS,
    canonical_field_name,
    torque_derivative_field,
)
from .motion import (
    dof_sorted_robot_joints,
    expand_coordinate_motion_by_robot_layout,
    infer_robot_model_dof,
    infer_robot_model_order,
)
from . import kots_api as kapi

Array = np.ndarray


@dataclass(frozen=True)
class TotalJointDynamicsStateRef:
    field: str
    refs: tuple[Any, ...]


class KotsAdapter:
    """RoboKots-specific state access, Jacobian, and fallback operations."""

    def __init__(self, builder: Any, *, state_type: Any) -> None:
        self.builder = builder
        self.model = builder.model
        self.state_type = state_type
        self._model_dof_cache: int | None = None
        self._model_order_cache: int | None = None

    def update_dynamics_if_available(self) -> bool:
        if not bool(getattr(self.builder, "_needs_dynamics_update", False)):
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

    def update_kinematics(self, q: Array) -> None:
        q_vec = np.asarray(q, dtype=float).reshape(-1)
        dof = self.model_dof()
        order = self.model_order()

        if q_vec.size == dof * order:
            motion = q_vec
        elif q_vec.size == dof:
            motion = self.expand_coordinate_motion(q_vec, dof=dof, order=order)
        else:
            raise ValueError(
                "KotsStateBuilder: unexpected q size. "
                f"Expected dof ({dof}) or dof*order ({dof * order}), got {q_vec.size}."
            )

        self.model.import_motions(motion)
        if bool(getattr(self.builder, "_needs_dynamics_update", False)):
            if self.update_dynamics_if_available():
                return

        self.model.kinematics()
        if bool(getattr(self.builder, "_needs_dynamics_update", False)):
            self.update_dynamics_if_available()

    def model_dof(self) -> int:
        if self._model_dof_cache is not None:
            return int(self._model_dof_cache)
        dof = infer_robot_model_dof(self.model)
        self._model_dof_cache = int(dof)
        return int(dof)

    def model_order(self) -> int:
        if self._model_order_cache is not None:
            return int(self._model_order_cache)
        order = infer_robot_model_order(self.model)
        self._model_order_cache = int(order)
        return int(order)

    def expand_coordinate_motion(self, q: Array, *, dof: int, order: int) -> Array:
        return expand_coordinate_motion_by_robot_layout(
            self.model,
            q,
            dof=dof,
            order=order,
            error_prefix="KotsStateBuilder",
        )

    @staticmethod
    def state_field_name(field: Any) -> str:
        return canonical_field_name(str(field))

    @staticmethod
    def fallback_backend_field_name(state_field: str) -> str:
        return kapi.fallback_backend_field_name(state_field)

    def make_state_type(
        self,
        *,
        owner_type: str,
        owner_name: str,
        state_field: str,
        frame_name: str | None,
    ) -> Any:
        return kapi.make_state_type(
            owner_type=owner_type,
            owner_name=owner_name,
            state_field=state_field,
            frame_name=frame_name,
            state_type=self.state_type,
        )

    @staticmethod
    def torque_derivative_order_from_state_data_type(data_type: str) -> int | None:
        return kapi.torque_derivative_order_from_state_data_type(data_type)

    def dof_sorted_joints(self) -> list[Any] | None:
        return dof_sorted_robot_joints(self.model)

    def total_joint_dynamics_frame_name(self, *, state_field: str, key: StateKey) -> str | None:
        frame_name = getattr(key, "frame", None) or "world"
        if "torque" in state_field:
            return None
        return str(frame_name)

    def resolve_total_joint_dynamics_refs(self, *, state_field: str, key: StateKey) -> tuple[Any, ...] | None:
        joints = self.dof_sorted_joints()
        if joints is None:
            return None
        if len(joints) == 0:
            return tuple()

        frame_name = self.total_joint_dynamics_frame_name(state_field=state_field, key=key)
        refs: list[Any] = []
        for joint in joints:
            joint_name = str(getattr(joint, "name", ""))
            if joint_name == "":
                raise ValueError("KotsStateBuilder: joint.name must be non-empty for dynamics expansion.")
            refs.append(
                self.make_state_type(
                    owner_type="joint",
                    owner_name=joint_name,
                    state_field=state_field,
                    frame_name=frame_name,
                )
            )
        return tuple(refs)

    @staticmethod
    def state_ref_data_type(state_ref: Any) -> str | None:
        return kapi.state_ref_data_type(state_ref)

    @staticmethod
    def state_ref_owner_type(state_ref: Any) -> str | None:
        return kapi.state_ref_owner_type(state_ref)

    @staticmethod
    def state_ref_owner_name(state_ref: Any) -> str | None:
        return kapi.state_ref_owner_name(state_ref)

    @staticmethod
    def state_ref_frame_name(state_ref: Any) -> str | None:
        return kapi.state_ref_frame_name(state_ref)

    def raise_missing_state_key(self, *, state_ref: Any, cause: KeyError) -> None:
        data_type = self.state_ref_data_type(state_ref)
        deriv_order = None if data_type is None else self.torque_derivative_order_from_state_data_type(data_type)
        if isinstance(deriv_order, int) and deriv_order > 0:
            field = torque_derivative_field(deriv_order)
            required_model_order = deriv_order + 3
            raise ValueError(
                "KotsStateBuilder: dynamics field "
                f"{field!r} requires RoboKots model order >= {required_model_order}. "
                f"Current model order is {self.model_order()}."
            ) from cause
        raise cause

    def value_from_single_state_ref(self, state_ref: Any) -> Array:
        try:
            return kapi.state_info(self.model, state_ref)
        except KeyError as e:
            self.raise_missing_state_key(state_ref=state_ref, cause=e)

    def value_from_state_ref(self, state_ref: Any) -> Array:
        total_joint_ref = self.as_total_joint_dynamics_state_ref(state_ref)
        if total_joint_ref is not None:
            return self.concat_total_joint_values(total_joint_ref.refs)
        return self.value_from_single_state_ref(state_ref)

    def value_from_state_refs(self, refs: tuple[Any, ...]) -> Array:
        if len(refs) == 0:
            return np.zeros((0,), dtype=float)

        values = kapi.state_info_list(self.model, refs)
        if values is not None:
            return values
        return self.concat_total_joint_values(refs)

    def jac_from_single_state_ref(self, state_ref: Any) -> Array:
        try:
            J = self.builder._jacobian_ops.dense(state_ref)
        except KeyError as e:
            self.raise_missing_state_key(state_ref=state_ref, cause=e)
        return self.normalize_jacobian_shape(J, state_ref=state_ref)

    def normalize_jacobian_shape(self, J_raw: Any, *, state_ref: Any) -> Array:
        J = np.asarray(J_raw, dtype=float)
        if J.ndim != 2:
            raise ValueError(f"Kots Jacobian must be 2D, got shape {J.shape}.")

        n_motion = int(self.model_dof() * self.model_order())
        if J.shape[1] == n_motion:
            return J
        if J.shape[0] == n_motion:
            return J.T

        total_joint_ref = self.as_total_joint_dynamics_state_ref(state_ref)
        if total_joint_ref is None:
            m = int(self.value_from_single_state_ref(state_ref).size)
        else:
            m = int(self.value_from_state_refs(total_joint_ref.refs).size)
        if J.shape[0] == m:
            return J
        if J.shape[1] == m:
            return J.T
        raise ValueError(f"Kots Jacobian must be ({m},n) or (n,{m}), got {J.shape}.")

    def jac_from_state_refs(self, refs: tuple[Any, ...]) -> Array:
        if len(refs) == 0:
            return np.zeros((0, self.model_dof() * self.model_order()), dtype=float)

        J = self.builder._jacobian_ops.dense_list(refs)
        if J is not None:
            return self.normalize_jacobian_shape(J, state_ref=TotalJointDynamicsStateRef(field="", refs=refs))
        return self.stack_total_joint_jacobians_fallback(refs)

    def matvec_from_single_state_ref(self, state_ref: Any, vec: Array) -> Array:
        return self.builder._jacobian_ops.jvp(state_ref, vec)

    def jac_from_matrix_mul_single_state_ref(self, state_ref: Any, cols: Array) -> Array:
        return self.builder._jacobian_ops.jvp(state_ref, cols)

    def jac_from_matvec_single_state_ref(self, state_ref: Any, cols: Array) -> Array:
        C = np.asarray(cols, dtype=float)
        if C.ndim != 2:
            raise ValueError(f"Kots matvec columns must be 2D, got shape {C.shape}.")
        try:
            m = int(self.value_from_single_state_ref(state_ref).size)
            return self.builder._jacobian_ops.jvp(state_ref, C, value_size=m)
        except KeyError as e:
            self.raise_missing_state_key(state_ref=state_ref, cause=e)

    def value(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del q, key
        return self.value_from_state_ref(state_ref)

    def jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del q
        total_joint_ref = self.as_total_joint_dynamics_state_ref(state_ref)
        if total_joint_ref is not None:
            return self.stack_total_joint_jacobians(total_joint_ref.refs)

        J = self.jac_from_single_state_ref(state_ref)
        return self.rotate_link_kinematics_jacobian_to_world(J=J, key=key, state_ref=state_ref)

    def link_world_rotation(self, *, state_ref: Any) -> Array | None:
        owner_type = self.state_ref_owner_type(state_ref)
        if owner_type != self.builder.owner_type:
            return None

        owner_name = self.state_ref_owner_name(state_ref)
        if not isinstance(owner_name, str) or owner_name == "":
            return None

        frame_name = self.state_ref_frame_name(state_ref) or "world"
        try:
            rot_ref = self.make_state_type(
                owner_type=owner_type,
                owner_name=owner_name,
                state_field="rot",
                frame_name=frame_name,
            )
            rot_flat = self.value_from_single_state_ref(rot_ref)
        except Exception:
            return None

        if int(rot_flat.size) != 9:
            return None
        return np.asarray(rot_flat, dtype=float).reshape(3, 3)

    def rotate_link_kinematics_jacobian_to_world(
        self,
        *,
        J: Array,
        key: StateKey,
        state_ref: Any,
    ) -> Array:
        if getattr(key, "dtype", None) != DTYPE_KINEMATICS:
            return J
        owner = getattr(key, "owner", None)
        if getattr(owner, "owner_type", None) != self.builder.owner_type:
            return J

        rot = self.link_world_rotation(state_ref=state_ref)
        if rot is None:
            return J

        rows = int(J.shape[0])
        if rows % 3 != 0:
            return J

        J_world = np.asarray(J, dtype=float).copy()
        for i in range(0, rows, 3):
            J_world[i : i + 3, :] = rot @ J_world[i : i + 3, :]
        return J_world

    def rotate_link_kinematics_rhs_to_local(
        self,
        *,
        rhs: Array,
        key: StateKey,
        state_ref: Any,
    ) -> Array:
        if getattr(key, "dtype", None) != DTYPE_KINEMATICS:
            return rhs
        owner = getattr(key, "owner", None)
        if getattr(owner, "owner_type", None) != self.builder.owner_type:
            return rhs

        rot = self.link_world_rotation(state_ref=state_ref)
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

    def transpose_matvec_from_single_state_ref(self, state_ref: Any, rhs: Array) -> Array:
        return self.builder._jacobian_ops.vjp(state_ref, rhs)

    def transpose_matvec_from_state_ref(self, state_ref: Any, rhs: Array) -> Array:
        total_joint_ref = self.as_total_joint_dynamics_state_ref(state_ref)
        R = np.asarray(rhs, dtype=float)
        if R.ndim not in (1, 2):
            raise ValueError(f"Kots jacobian_transpose_mul rhs must be 1D or 2D, got shape {R.shape}.")

        if total_joint_ref is None:
            return self.transpose_matvec_from_single_state_ref(state_ref, R)

        try:
            return np.asarray(self.transpose_matvec_from_single_state_ref(list(total_joint_ref.refs), R), dtype=float)
        except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
            pass

        parts: list[Array] = []
        offset = 0
        for ref in total_joint_ref.refs:
            m = int(self.value_from_single_state_ref(ref).size)
            stop = offset + m
            if stop > int(R.shape[0]):
                raise ValueError(
                    "KotsStateBuilder: rhs is too short for total_joint transpose multiply. "
                    f"Need at least {stop} rows, got {R.shape[0]}."
                )
            parts.append(
                np.asarray(
                    self.transpose_matvec_from_single_state_ref(ref, R[offset:stop, ...]),
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
            return np.zeros((self.model_dof() * self.model_order(),), dtype=float)
        return np.sum(np.stack(parts, axis=0), axis=0)

    @staticmethod
    def as_total_joint_dynamics_state_ref(state_ref: Any) -> TotalJointDynamicsStateRef | None:
        if isinstance(state_ref, TotalJointDynamicsStateRef):
            return state_ref
        if (
            isinstance(state_ref, tuple)
            and len(state_ref) == 3
            and state_ref[0] == "total_joint_dynamics"
        ):
            return TotalJointDynamicsStateRef(field=str(state_ref[1]), refs=tuple(state_ref[2]))
        return None

    def concat_total_joint_values(self, refs: tuple[Any, ...]) -> Array:
        parts = [self.value_from_single_state_ref(ref) for ref in refs]
        if len(parts) == 0:
            return np.zeros((0,), dtype=float)
        return np.concatenate(parts, axis=0)

    def stack_total_joint_jacobians(self, refs: tuple[Any, ...]) -> Array:
        return self.jac_from_state_refs(refs)

    def stack_total_joint_jacobians_fallback(self, refs: tuple[Any, ...]) -> Array:
        blocks = [self.jac_from_single_state_ref(ref) for ref in refs]
        if len(blocks) == 0:
            return np.zeros((0, self.model_dof() * self.model_order()), dtype=float)
        ncols = int(blocks[0].shape[1])
        for block in blocks[1:]:
            if int(block.shape[1]) != ncols:
                raise ValueError(
                    "KotsStateBuilder: inconsistent Jacobian column size while stacking total_joint dynamics. "
                    f"Expected {ncols}, got {block.shape[1]}."
                )
        return np.vstack(blocks)


__all__ = [
    "KotsAdapter",
    "TotalJointDynamicsStateRef",
]
