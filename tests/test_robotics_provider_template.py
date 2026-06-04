from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rei.backends.state.robotics.provider import (
    RobotFieldBinding,
    RobotFieldHandler,
    RobotStateRef,
    TrajectoryRoboticsStateProvider,
    RoboticsStateProvider,
    assert_provider_contract,
    assert_trajectory_provider_contract,
    robot_field_bindings_from_names,
)
from rei.core.state_schema import DTYPE_COORD, DTYPE_KINEMATICS, make_jac_key, make_key
from rei.core.trajectory import TrajectoryMap


def _pos_value_handler(q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
    del key
    q_vec = np.asarray(q, dtype=float).reshape(-1)
    offset = float(state_ref.get("offset", 0.0))
    return np.array([q_vec[0] + offset, q_vec[1], q_vec.sum()], dtype=float)


def _pos_jac_handler(q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
    del key, state_ref
    n = int(np.asarray(q, dtype=float).reshape(-1).size)
    J = np.zeros((3, n), dtype=float)
    J[0, 0] = 1.0
    J[1, 1] = 1.0
    J[2, :] = 1.0
    return J


class _TableBackend:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def update(self, q: np.ndarray, model: dict[str, Any], data: dict[str, Any]) -> None:
        del model
        self.calls.append(q.copy())
        data["last_q"] = q.copy()

    def ref(self, key: Any, model: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        del model, data
        return {"owner_name": key.owner.owner_name, "field": key.field, "offset": 4.0}

    def pos(self, q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
        return _pos_value_handler(q, key, state_ref)

    def pos_jac(self, q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
        return _pos_jac_handler(q, key, state_ref)


class TestRoboticsProviderTemplate:
    def test_robotics_state_provider_registers_custom_kinematics_handlers(self) -> None:
        calls: list[np.ndarray] = []

        def _update_model(q: np.ndarray, model: dict[str, Any], data: dict[str, Any]) -> None:
            calls.append(q.copy())
            data["last_q"] = q.copy()
            model["updated"] = True

        def _resolve_state_ref(key: Any, model: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
            del model, data
            return {
                "owner_name": key.owner.owner_name,
                "field": key.field,
                "offset": 10.0,
            }

        provider = RoboticsStateProvider(
            model={},
            data={},
            q_var="q",
            kinematics_field_handlers={
                "pos": RobotFieldHandler(
                    value_handler=_pos_value_handler,
                    jac_handler=_pos_jac_handler,
                )
            },
            update_model=_update_model,
            resolve_state_ref=_resolve_state_ref,
        )

        x = np.array([1.0, 2.0], dtype=float)
        key_v = make_key(
            k=0,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
        )
        key_j = make_jac_key(
            k=0,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
            var="q",
        )

        assert provider.accepts(key_v)
        assert provider.accepts(key_j)

        out = provider.build_state(x, required=[key_v, key_j])
        assert set(out.keys()) == {key_v, key_j}
        assert np.allclose(out[key_v], np.array([11.0, 2.0, 3.0], dtype=float))
        assert np.allclose(out[key_j], np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float))
        assert len(calls) == 1
        assert np.allclose(calls[0], x)

    def test_robotics_state_provider_can_provide_total_joint_q(self) -> None:
        provider = RoboticsStateProvider(
            model={},
            data={},
            register_joint_q=True,
        )
        x = np.array([0.5, -1.0, 2.0], dtype=float)
        key_v = make_key(
            k=0,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_COORD,
            field="q",
        )
        key_j = make_jac_key(
            k=0,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_COORD,
            field="q",
            var="q",
        )

        out = provider.build_state(x, required=[key_v, key_j])
        assert np.allclose(out[key_v], x)
        assert np.allclose(out[key_j], np.eye(3, dtype=float))

    def test_robotics_state_provider_requires_some_registered_field(self) -> None:
        with pytest.raises(ValueError):
            _ = RoboticsStateProvider(
                model={},
                data={},
                register_joint_q=False,
            )

    def test_robotics_state_provider_default_state_ref_is_typed(self) -> None:
        provider = RoboticsStateProvider(
            model={},
            data={},
            register_joint_q=False,
            kinematics_field_handlers={
                "pos": RobotFieldHandler(
                    value_handler=lambda q, key, state_ref: np.asarray([state_ref.k, q[0]], dtype=float),
                )
            },
        )
        key = make_key(
            k=0,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
        )
        state_ref = provider._state_ref(key)
        assert isinstance(state_ref, RobotStateRef)
        assert state_ref.owner_name == "ee"
        assert state_ref.field == "pos"

    def test_robotics_state_provider_reports_jacobian_row_mismatch(self) -> None:
        provider = RoboticsStateProvider(
            model={},
            data={},
            register_joint_q=False,
            kinematics_field_handlers={
                "pos": RobotFieldHandler(
                    value_handler=lambda q, key, state_ref: np.array([1.0, 2.0], dtype=float),
                    jac_handler=lambda q, key, state_ref: np.ones((3, 2), dtype=float),
                )
            },
        )
        key = make_jac_key(
            k=0,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
            var="q",
        )
        with pytest.raises(ValueError, match="row mismatch") as exc_info:
            provider.build_state(np.array([1.0, 2.0], dtype=float), required=[key])
        message = str(exc_info.value)
        assert "handler=jac_handler" in message
        assert "dtype='kinematics'" in message
        assert "owner_name='ee'" in message
        assert "field='pos_J_q'" in message
        assert "Expected shape=(2, *)" in message
        assert "actual shape=(3, 2)" in message

    def test_robotics_state_provider_reports_jacobian_column_mismatch(self) -> None:
        provider = RoboticsStateProvider(
            model={},
            data={},
            register_joint_q=False,
            kinematics_field_handlers={
                "pos": RobotFieldHandler(
                    value_handler=lambda q, key, state_ref: np.array([1.0, 2.0], dtype=float),
                    jac_handler=lambda q, key, state_ref: np.ones((2, 3), dtype=float),
                )
            },
        )
        key = make_jac_key(
            k=0,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
            var="q",
        )
        with pytest.raises(ValueError, match="column mismatch") as exc_info:
            provider.build_state(np.array([1.0, 2.0], dtype=float), required=[key])
        message = str(exc_info.value)
        assert "handler=jac_handler" in message
        assert "jacobian_wrt='q'" in message
        assert "Expected shape=(2, 2)" in message
        assert "actual shape=(2, 3)" in message

    def test_robotics_state_provider_reports_non_2d_jacobian_context(self) -> None:
        provider = RoboticsStateProvider(
            model={},
            data={},
            register_joint_q=False,
            kinematics_field_handlers={
                "pos": RobotFieldHandler(
                    value_handler=lambda q, key, state_ref: np.array([1.0, 2.0], dtype=float),
                    jac_handler=lambda q, key, state_ref: np.ones(2, dtype=float),
                )
            },
        )
        key = make_jac_key(
            k=0,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
            var="q",
        )
        with pytest.raises(ValueError, match="Expected shape=\\(m, n\\)") as exc_info:
            provider.build_state(np.array([1.0, 2.0], dtype=float), required=[key])
        message = str(exc_info.value)
        assert "handler=jac_handler" in message
        assert "actual shape=(2,)" in message

    def test_robotics_state_provider_rejects_unknown_jacobian_wrt(self) -> None:
        with pytest.raises(ValueError, match="unsupported jacobian_wrt") as exc_info:
            RoboticsStateProvider(
                model={},
                data={},
                register_joint_q=False,
                kinematics_field_handlers={
                    "pos": RobotFieldHandler(
                        value_handler=lambda q, key, state_ref: np.array([1.0], dtype=float),
                        jac_handler=lambda q, key, state_ref: np.ones((1, 2), dtype=float),
                        jacobian_wrt="qq",
                    )
                },
            )
        message = str(exc_info.value)
        assert "field='pos'" in message
        assert "expected 'q' or 'state'" in message

    def test_robotics_state_provider_can_be_built_from_field_bindings(self) -> None:
        adapter = _TableBackend()
        data: dict[str, Any] = {}
        provider = RoboticsStateProvider.from_field_bindings(
            model={},
            data=data,
            handler_owner=adapter,
            update_model="update",
            resolve_state_ref="ref",
            register_joint_q=False,
            field_bindings=[
                RobotFieldBinding(
                    dtype=DTYPE_KINEMATICS,
                    owner_type="link",
                    field="pos",
                    value="pos",
                    jac="pos_jac",
                )
            ],
        )
        key_v = make_key(
            k=0,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
        )
        key_j = make_jac_key(
            k=0,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
            var="q",
        )

        out = provider.build_state(np.array([1.0, 2.0], dtype=float), required=[key_v, key_j])

        assert np.allclose(out[key_v], np.array([5.0, 2.0, 3.0], dtype=float))
        assert np.allclose(out[key_j], np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float))
        assert len(adapter.calls) == 1
        assert np.allclose(data["last_q"], np.array([1.0, 2.0], dtype=float))

    def test_robotics_state_provider_field_bindings_report_missing_method(self) -> None:
        with pytest.raises(ValueError, match="missing_pos"):
            RoboticsStateProvider.from_field_bindings(
                model={},
                data={},
                handler_owner=_TableBackend(),
                register_joint_q=False,
                field_bindings=[
                    RobotFieldBinding(
                        dtype=DTYPE_KINEMATICS,
                        owner_type="link",
                        field="pos",
                        value="missing_pos",
                    )
                ],
            )

    def test_robotics_state_provider_can_be_built_from_name_bindings(self) -> None:
        adapter = _TableBackend()
        provider = RoboticsStateProvider.from_name_bindings(
            model={},
            data={},
            handler_owner=adapter,
            update_model="update",
            resolve_state_ref="ref",
            register_joint_q=False,
            kinematics_link_pos="pos",
            kinematics_link_pos_J_q="pos_jac",
        )
        key_v = make_key(
            k=0,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
        )
        key_j = make_jac_key(
            k=0,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
            var="q",
        )

        out = provider.build_state(np.array([1.0, 2.0], dtype=float), required=[key_v, key_j])

        assert np.allclose(out[key_v], np.array([5.0, 2.0, 3.0], dtype=float))
        assert np.allclose(out[key_j], np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float))

    def test_robot_field_bindings_from_names_supports_total_joint_owner_type(self) -> None:
        bindings = robot_field_bindings_from_names(
            {
                "dynamics_total_joint_torque": "torque",
                "dynamics_total_joint_torque_J_state": "torque_jac",
            }
        )
        assert len(bindings) == 1
        binding = bindings[0]
        assert binding.dtype == "dynamics"
        assert binding.owner_type == "total_joint"
        assert binding.field == "torque"
        assert binding.value == "torque"
        assert binding.jac == "torque_jac"
        assert binding.jacobian_wrt == "state"

    def test_robot_field_bindings_from_names_requires_value_binding(self) -> None:
        with pytest.raises(ValueError, match="Missing value"):
            robot_field_bindings_from_names({"kinematics_link_pos_J_q": "pos_jac"})

    def test_assert_provider_contract_accepts_expected_fields_and_shapes(self) -> None:
        provider = RoboticsStateProvider(
            model={},
            data={},
            register_joint_q=False,
            kinematics_field_handlers={
                "pos": RobotFieldHandler(
                    value_handler=_pos_value_handler,
                    jac_handler=_pos_jac_handler,
                )
            },
        )
        key_v = make_key(
            k=0,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
        )
        key_j = make_jac_key(
            k=0,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
            var="q",
        )

        out = assert_provider_contract(
            provider,
            np.array([1.0, 2.0], dtype=float),
            [key_v, key_j],
            expected_shapes={key_v: (3,), key_j: (3, 2)},
        )
        assert np.allclose(out[key_v], np.array([1.0, 2.0, 3.0], dtype=float))
        assert np.allclose(out[key_j], np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float))

    def test_assert_provider_contract_reports_shape_mismatch(self) -> None:
        provider = RoboticsStateProvider(
            model={},
            data={},
            register_joint_q=False,
            kinematics_field_handlers={
                "pos": RobotFieldHandler(
                    value_handler=_pos_value_handler,
                    jac_handler=_pos_jac_handler,
                )
            },
        )
        key = make_key(
            k=0,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
        )
        with pytest.raises(AssertionError, match="shape mismatch") as exc_info:
            assert_provider_contract(
                provider,
                np.array([1.0, 2.0], dtype=float),
                [key],
                expected_shapes={key: (2,)},
            )
        assert "owner_name='ee'" in str(exc_info.value)


class TestTrajectoryRoboticsProviderTemplate:
    def test_trajectory_provider_chains_kinematics_jacobian_to_parameters(self) -> None:
        traj = TrajectoryMap.from_blocks(
            [
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
                np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
            ]
        )
        calls: list[np.ndarray] = []

        def _update_model(q: np.ndarray, model: dict[str, Any], data: dict[str, Any]) -> None:
            del model
            calls.append(q.copy())
            data["last_q"] = q.copy()

        provider = TrajectoryRoboticsStateProvider(
            model={},
            data={},
            trajectory_map=traj,
            p_var="p",
            kinematics_field_handlers={
                "pos": RobotFieldHandler(
                    value_handler=_pos_value_handler,
                    jac_handler=_pos_jac_handler,
                )
            },
            update_model=_update_model,
        )

        p = np.array([1.0, 2.0, 3.0], dtype=float)
        key_q = make_key(
            k=1,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_COORD,
            field="q",
        )
        key_q_j = make_jac_key(
            k=1,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_COORD,
            field="q",
            var="p",
        )
        key_pos_j = make_jac_key(
            k=1,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
            var="p",
        )

        out = provider.build_state(p, required=[key_q, key_q_j, key_pos_j])
        q_expected = np.array([2.0, 3.0], dtype=float)
        dqdp_expected = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        J_state = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)

        assert np.allclose(out[key_q], q_expected)
        assert np.allclose(out[key_q_j], dqdp_expected)
        assert np.allclose(out[key_pos_j], J_state @ dqdp_expected)
        assert len(calls) == 1
        assert np.allclose(calls[0], q_expected)

    def test_assert_trajectory_provider_contract_accepts_expected_keys(self) -> None:
        traj = TrajectoryMap.from_blocks(
            [
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
                np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
            ]
        )
        provider = TrajectoryRoboticsStateProvider(
            model={},
            data={},
            trajectory_map=traj,
            p_var="p",
            kinematics_field_handlers={
                "pos": RobotFieldHandler(
                    value_handler=_pos_value_handler,
                    jac_handler=_pos_jac_handler,
                )
            },
        )
        key_q = make_key(
            k=1,
            owner_type="total_joint",
            owner_name="robot",
            dtype=DTYPE_COORD,
            field="q",
        )
        key_pos_j = make_jac_key(
            k=1,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
            var="p",
        )

        out = assert_trajectory_provider_contract(
            provider,
            np.array([1.0, 2.0, 3.0], dtype=float),
            [key_q, key_pos_j],
            expected_shapes={key_q: (2,), key_pos_j: (3, 3)},
        )
        assert np.allclose(out[key_q], np.array([2.0, 3.0], dtype=float))

    def test_trajectory_provider_can_be_built_from_field_bindings(self) -> None:
        traj = TrajectoryMap.from_blocks(
            [
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
                np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
            ]
        )
        adapter = _TableBackend()
        provider = TrajectoryRoboticsStateProvider.from_field_bindings(
            model={},
            data={},
            trajectory_map=traj,
            handler_owner=adapter,
            update_model="update",
            resolve_state_ref="ref",
            p_var="p",
            register_joint_q=False,
            field_bindings=[
                RobotFieldBinding(
                    dtype=DTYPE_KINEMATICS,
                    owner_type="link",
                    field="pos",
                    value="pos",
                    jac="pos_jac",
                )
            ],
        )
        key = make_jac_key(
            k=1,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
            var="p",
        )

        out = provider.build_state(np.array([1.0, 2.0, 3.0], dtype=float), required=[key])

        assert np.allclose(out[key], np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=float))
        assert len(adapter.calls) == 1
        assert np.allclose(adapter.calls[0], np.array([2.0, 3.0], dtype=float))

    def test_trajectory_provider_can_be_built_from_name_bindings(self) -> None:
        traj = TrajectoryMap.from_blocks(
            [
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
                np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
            ]
        )
        adapter = _TableBackend()
        provider = TrajectoryRoboticsStateProvider.from_name_bindings(
            model={},
            data={},
            trajectory_map=traj,
            handler_owner=adapter,
            update_model="update",
            resolve_state_ref="ref",
            p_var="p",
            register_joint_q=False,
            kinematics_link_pos="pos",
            kinematics_link_pos_J_state="pos_jac",
        )
        key = make_jac_key(
            k=1,
            owner_type="link",
            owner_name="ee",
            dtype=DTYPE_KINEMATICS,
            field="pos",
            var="p",
        )

        out = provider.build_state(np.array([1.0, 2.0, 3.0], dtype=float), required=[key])

        assert np.allclose(out[key], np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=float))

    def test_trajectory_provider_can_chain_stacked_motion_jacobian(self) -> None:
        q_map = TrajectoryMap.from_blocks(
            [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)]
        )
        dq_map = TrajectoryMap.from_blocks(
            [np.array([[2.0, 0.0], [0.0, 3.0]], dtype=float)]
        )
        motions: list[np.ndarray] = []

        def _dyn_value(q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
            del q, key, state_ref
            return np.array([0.0], dtype=float)

        def _dyn_jac(q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
            del q, key, state_ref
            return np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float)

        def _update_motion(q: np.ndarray, motion: np.ndarray, k: int, model: Any, data: Any) -> None:
            del q, k, model, data
            motions.append(motion.copy())

        provider = TrajectoryRoboticsStateProvider(
            model={},
            data={},
            trajectory_map=q_map,
            trajectory_derivative_maps={1: dq_map},
            p_var="p",
            dynamics_field_handlers={
                "torque": RobotFieldHandler(
                    value_handler=_dyn_value,
                    jac_handler=_dyn_jac,
                )
            },
            update_motion_model=_update_motion,
            motion_layout="stacked",
            derivative_orders=(0, 1),
        )

        p = np.array([5.0, 7.0], dtype=float)
        key = make_jac_key(
            k=0,
            owner_type="total_joint",
            owner_name="robot",
            dtype="dynamics",
            field="torque",
            var="p",
        )

        out = provider.build_state(p, required=[key])
        dmotiondp = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 3.0],
            ],
            dtype=float,
        )
        assert np.allclose(out[key], np.array([[7.0, 14.0]], dtype=float))
        assert len(motions) == 1
        assert np.allclose(motions[0], np.array([5.0, 7.0, 10.0, 21.0], dtype=float))
