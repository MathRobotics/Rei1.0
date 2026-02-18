from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from eiopt.backends.state.vision import CameraCalibrationStateProvider, VisionFieldHandler
from eiopt.core.state_schema import DTYPE_VISION, vision_jac_key, vision_key


def _value_handler(q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
    del state_ref
    q_sum = float(np.asarray(q, dtype=float).reshape(-1).sum())
    return np.array([float(getattr(key, "k", 0)), q_sum], dtype=float)


def _jac_handler(q: np.ndarray, key: Any, state_ref: Any) -> np.ndarray:
    del key, state_ref
    n = int(np.asarray(q, dtype=float).reshape(-1).size)
    return np.ones((2, n), dtype=float)


class TestVisionProviderTemplate(unittest.TestCase):
    def test_vision_key_helpers_build_dtype_vision_keys(self) -> None:
        key_v = vision_key(k=3, owner_name="cam0", field="reproj")
        key_j = vision_jac_key(k=3, owner_name="cam0", field="reproj", var="theta")

        self.assertEqual(key_v.dtype, DTYPE_VISION)
        self.assertEqual(key_v.owner.owner_type, "camera")
        self.assertEqual(key_v.owner.owner_name, "cam0")
        self.assertEqual(key_v.field, "reproj")
        self.assertEqual(key_j.field, "reproj_J_theta")
        self.assertEqual(key_j.k, 3)

    def test_camera_calibration_state_provider_handles_nonzero_k(self) -> None:
        calls: list[np.ndarray] = []

        def _update_model(q: np.ndarray, model: Any, data: dict[str, Any]) -> None:
            del model
            calls.append(np.asarray(q, dtype=float).copy())
            data["last_q"] = q.copy()

        provider = CameraCalibrationStateProvider(
            model={"name": "cam-calib"},
            data={},
            param_var="theta",
            field_handlers={
                "reproj": VisionFieldHandler(
                    value_handler=_value_handler,
                    jac_handler=_jac_handler,
                )
            },
            update_model=_update_model,
            allow_nonzero_k=True,
        )

        x = np.array([1.0, 2.0, 3.0], dtype=float)
        key_v = vision_key(k=2, owner_name="cam0", field="reproj")
        key_j = vision_jac_key(k=2, owner_name="cam0", field="reproj", var="theta")

        self.assertTrue(provider.accepts(key_v))
        self.assertTrue(provider.accepts(key_j))

        out = provider.build_state(x, required=[key_v, key_j])
        self.assertEqual(set(out.keys()), {key_v, key_j})
        self.assertTrue(np.allclose(out[key_v], np.array([2.0, 6.0], dtype=float)))
        self.assertEqual(out[key_j].shape, (2, 3))
        self.assertEqual(len(calls), 1)
        self.assertTrue(np.allclose(calls[0], x))

    def test_camera_calibration_state_provider_requires_field_handlers(self) -> None:
        with self.assertRaises(ValueError):
            _ = CameraCalibrationStateProvider(
                model={},
                data={},
                field_handlers=None,
            )


if __name__ == "__main__":
    unittest.main()
