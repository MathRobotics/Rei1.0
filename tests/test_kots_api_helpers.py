from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


def _ensure_robokots_state_stub() -> None:
    try:
        importlib.import_module("robokots.core.state")
        return
    except ModuleNotFoundError as exc:
        if exc.name not in {"robokots", "robokots.core", "robokots.core.state"}:
            raise
    robokots_mod = types.ModuleType("robokots")
    core_mod = types.ModuleType("robokots.core")
    state_mod = types.ModuleType("robokots.core.state")

    class StateType:
        def __init__(self, owner_type: str, owner_name: str, field: str, frame: str | None) -> None:
            self.owner_type = owner_type
            self.owner_name = owner_name
            self.field = field
            self.frame = frame

    state_mod.StateType = StateType
    core_mod.state = state_mod
    robokots_mod.core = core_mod

    sys.modules["robokots"] = robokots_mod
    sys.modules["robokots.core"] = core_mod
    sys.modules["robokots.core.state"] = state_mod


_ensure_robokots_state_stub()
kots_api = importlib.import_module("rei.backends.state.robotics.kots_api")
optional = importlib.import_module("rei.backends.optional")


@pytest.mark.parametrize("method", ["numerical", "autodiff"])
def test_explicit_jacobian_method_never_uses_analytic(method):
    if method == "autodiff":
        pytest.importorskip("jax")
    matrix = np.array([[1., 2., 3.], [4., 5., 6.]])

    class Model:
        def jacobian(self, ref, *, numerical=False):
            assert numerical
            return matrix

        def jacobian_mul(self, ref, rhs, *, numerical=False):
            assert numerical
            return matrix @ rhs

        def jacobian_transpose_mul(self, ref, rhs, *, numerical=False):
            assert numerical
            return matrix.T @ rhs

        def jacobian_autodiff(self, ref):
            return matrix

    op = kots_api.RoboKotsJacobianOperator(Model(), jacobian_method=method)
    np.testing.assert_array_equal(op.dense(object()), matrix)
    np.testing.assert_array_equal(op.dense_list((object(),)), matrix)
    for cols in (np.ones(3), np.eye(3)):
        np.testing.assert_array_equal(op.jvp(object(), cols), matrix @ cols)
    for rhs in (np.ones(2), np.eye(2)):
        np.testing.assert_array_equal(op.vjp(object(), rhs), matrix.T @ rhs)


def test_invalid_jacobian_method_is_rejected():
    with pytest.raises(ValueError, match="jacobian_method"):
        kots_api.RoboKotsJacobianOperator(object(), jacobian_method="typo")


@pytest.mark.parametrize("method", ["numerical", "autodiff"])
def test_unsupported_derivative_does_not_fall_back(method):
    if method == "autodiff":
        pytest.importorskip("jax")

    class Model:
        def jacobian(self, ref, *, numerical=False):
            assert numerical, "must not fall back to analytic"
            raise NotImplementedError("unsupported state")

        def jacobian_autodiff(self, ref):
            raise NotImplementedError("unsupported state")

    op = kots_api.RoboKotsJacobianOperator(Model(), jacobian_method=method)
    with pytest.raises(NotImplementedError, match="unsupported state"):
        op.dense_list((object(),))


def test_autodiff_uses_local_double_precision():
    jax = pytest.importorskip("jax")

    class Model:
        def jacobian_autodiff(self, ref):
            assert jax.config.x64_enabled
            return np.eye(2)

    before = jax.config.x64_enabled
    kots_api.RoboKotsJacobianOperator(Model(), jacobian_method="autodiff").dense(object())
    assert jax.config.x64_enabled == before


class _StrictStateType:
    def __init__(self, owner_type: str, owner_name: str, field: str, frame: str | None) -> None:
        if field == "torque_d1":
            raise KeyError(field)
        self.owner_type = owner_type
        self.owner_name = owner_name
        self.field = field
        self.frame = frame


class _MulModel:
    def jacobian_mul(self, state_ref, cols):
        if isinstance(state_ref, np.ndarray):
            raise TypeError("expects state_ref first")
        return np.asarray(cols, dtype=float).T


class _VecMulModel:
    def matvec(self, vec, state_ref):
        del state_ref
        return np.asarray(vec, dtype=float)[:2]


def test_make_state_type_falls_back_to_robokots_torque_diff_alias() -> None:
    ref = kots_api.make_state_type(
        owner_type="joint",
        owner_name="j0",
        state_field="torque_d1",
        frame_name=None,
        state_type=_StrictStateType,
    )
    assert ref.field == "torque_diff1"


def test_jacobian_matrix_mul_normalizes_transposed_output() -> None:
    cols = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    out = kots_api.jacobian_matrix_mul(_MulModel(), object(), cols)
    assert np.allclose(out, cols)


def test_jacobian_from_mul_falls_back_to_vector_matvec() -> None:
    cols = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
    out = kots_api.jacobian_from_mul(_VecMulModel(), object(), cols, value_size=2)
    assert np.allclose(out, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))


def test_robokots_jacobian_operator_exposes_dense_jvp_vjp_facade() -> None:
    class Model:
        def jacobian(self, state_ref):
            del state_ref
            return np.eye(2, dtype=float)

        def jacobian_mul(self, state_ref, cols):
            del state_ref
            return np.asarray(cols, dtype=float)

        def jacobian_transpose_mul(self, state_ref, rhs):
            del state_ref
            return np.asarray(rhs, dtype=float)

    ops = kots_api.RoboKotsJacobianOperator(Model())
    ref = object()
    cols = np.eye(2, dtype=float)
    rhs = np.array([1.0, 2.0], dtype=float)

    assert np.allclose(ops.dense(ref), np.eye(2, dtype=float))
    assert np.allclose(ops.jvp(ref, cols), cols)
    assert np.allclose(ops.vjp(ref, rhs), rhs)


def test_robokots_jacobian_operator_matrix_jvp_falls_back_to_vector_matvec() -> None:
    cols = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
    out = kots_api.RoboKotsJacobianOperator(_VecMulModel()).jvp(object(), cols)
    assert np.allclose(out, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))


def test_optional_backend_import_error_mentions_uv_group() -> None:
    with pytest.raises(ImportError, match="uv sync --group missing-backend"):
        optional.import_optional_backend(
            "_rei_missing_backend_module_for_test",
            backend_name="rei.test.backend",
            install_hint="uv sync --group missing-backend",
        )
