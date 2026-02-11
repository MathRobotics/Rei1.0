from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.state_cache import StateKey
from ..core.state_schema import DTYPE_KINEMATICS, split_jac_field
from ..core.trajectory import TrajectoryMap
from ._template import BackendDispatchStateBuilder

try:
    from robokots.core.state import StateType
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "`eiopt.backends.kots` requires the robotics RoboKots bindings. "
        "Install RoboKots (e.g. via github) and retry."
    ) from e

Array = np.ndarray


@dataclass(frozen=True)
class KotsFieldFamily:
    field: str


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
    ) -> None:
        super().__init__(model, data, q_var=q_var)
        self.dtype = DTYPE_KINEMATICS
        self.owner_type = "link"

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
            )
            self.field_to_jac[spec.field] = jac_name

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
        self.model.kinematics()

    def _model_dof(self) -> int:
        dof_fn = getattr(self.model, "dof", None)
        if callable(dof_fn):
            return int(dof_fn())
        robot = getattr(self.model, "robot_", None)
        if robot is not None and hasattr(robot, "dof"):
            return int(getattr(robot, "dof"))
        raise ValueError("KotsStateBuilder: unable to resolve model dof.")

    def _model_order(self) -> int:
        order_fn = getattr(self.model, "order", None)
        if callable(order_fn):
            order = int(order_fn())
        else:
            order = int(getattr(self.model, "order_", 1))
        if order < 1:
            raise ValueError(f"KotsStateBuilder: model order must be >= 1, got {order}.")
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
        if owner_type != self.owner_type or not isinstance(owner_name, str) or owner_name == "":
            raise ValueError(
                f"Kots backend expects owner_type={self.owner_type!r} in key, got: {key!r}"
            )
        frame_name = getattr(key, "frame", None) or "world"
        return StateType(self.owner_type, owner_name, key.field, str(frame_name))

    def _value_from_state_ref(self, state_ref: Any) -> Array:
        return np.asarray(self.model.state_info(state_ref), dtype=float).reshape(-1)

    def _handle_value(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del q, key
        return self._value_from_state_ref(state_ref)

    def _handle_jac(self, q: Array, key: StateKey, state_ref: Any) -> Array:
        del key
        del q
        J = np.asarray(self.model.jacobian(state_ref), dtype=float)
        if J.ndim != 2:
            raise ValueError(f"Kots Jacobian must be 2D, got shape {J.shape}.")

        m = int(self._value_from_state_ref(state_ref).size)
        if J.shape[0] == m:
            return J
        if J.shape[1] == m:
            return J.T
        raise ValueError(f"Kots Jacobian must be ({m},n) or (n,{m}), got {J.shape}.")


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
        p_var: str = "p",
        fields: Sequence[str] | None = None,
    ) -> None:
        self.trajectory_map = trajectory_map
        super().__init__(model, data, q_var=p_var, fields=fields)

    @classmethod
    def from_dsl(
        cls,
        model: Any,
        data: Any,
        *,
        dsl: Mapping[str, Any],
        trajectory_key: str = "trajectory",
        fields: Sequence[str] | None = None,
        sync_var_dim: bool = True,
    ) -> "KotsTrajectoryStateBuilder":
        if not isinstance(dsl, Mapping):
            raise TypeError("KotsTrajectoryStateBuilder.from_dsl: dsl must be a mapping.")
        traj_dsl = dsl.get(trajectory_key, None)
        if not isinstance(traj_dsl, Mapping):
            raise ValueError(
                "KotsTrajectoryStateBuilder.from_dsl: trajectory section is required. "
                f"Expected mapping at dsl[{trajectory_key!r}]."
            )

        p_var = str(traj_dsl.get("var", "p"))
        if p_var == "":
            raise ValueError("KotsTrajectoryStateBuilder.from_dsl: trajectory.var must be non-empty.")

        default_steps = None
        time_dsl = dsl.get("time", None)
        if isinstance(time_dsl, Mapping) and "N" in time_dsl:
            default_steps = int(time_dsl["N"]) + 1

        default_q_dim = cls._infer_model_dof(model)
        trajectory_map = TrajectoryMap.from_dsl(
            traj_dsl,
            default_steps=default_steps,
            default_q_dim=default_q_dim,
        )
        if sync_var_dim:
            cls._sync_var_dim_from_dsl(dsl, var_name=p_var, dim=trajectory_map.p_dim)
        return cls(
            model,
            data,
            trajectory_map=trajectory_map,
            p_var=p_var,
            fields=fields,
        )

    @staticmethod
    def _infer_model_dof(model: Any) -> int | None:
        dof_fn = getattr(model, "dof", None)
        if callable(dof_fn):
            try:
                return int(dof_fn())
            except Exception:
                return None
        robot = getattr(model, "robot_", None)
        if robot is not None and hasattr(robot, "dof"):
            try:
                return int(getattr(robot, "dof"))
            except Exception:
                return None
        return None

    @staticmethod
    def _sync_var_dim_from_dsl(dsl: Mapping[str, Any], *, var_name: str, dim: int) -> None:
        if not isinstance(dsl, dict):
            return
        variables = dsl.get("variables", None)
        if not isinstance(variables, list):
            return
        for v in variables:
            if not isinstance(v, dict):
                continue
            if v.get("name", None) != var_name:
                continue
            v["dim"] = int(dim)
            init = np.asarray(v.get("init", []), dtype=float).reshape(-1)
            if init.size != int(dim):
                v["init"] = np.zeros((int(dim),), dtype=float).tolist()
            return

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
            self._update_kinematics(q_k)

            for key, entry in grouped[k]:
                state_ref = self._state_ref(key, state_ref_field=entry.state_ref_field)
                value = entry.handler(q_k, key, state_ref)

                if self._is_param_jac_key(key):
                    J_q = np.asarray(value, dtype=float)
                    if J_q.ndim != 2:
                        raise ValueError(
                            f"KotsTrajectoryStateBuilder: Jacobian must be 2D, got shape {J_q.shape} for key {key!r}."
                        )
                    if J_q.shape[1] != dqdp_k.shape[0]:
                        raise ValueError(
                            "KotsTrajectoryStateBuilder: Jacobian chain mismatch. "
                            f"J_q has shape {J_q.shape}, dqdp has shape {dqdp_k.shape}."
                        )
                    value = J_q @ dqdp_k

                out[key] = value

        return out
