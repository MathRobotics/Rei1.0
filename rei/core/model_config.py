"""Backend-independent model loading metadata for problem specifications."""
from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any


def normalize_model_config(config: Any, *, base_dir: Path | None = None) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("model must be a TOML table")
    unknown = config.keys() - {"file", "locked_joints"}
    if unknown:
        raise ValueError(f"Unknown model configuration keys: {sorted(unknown)}")
    result: dict[str, Any] = {}
    if "file" in config:
        filename = config["file"]
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("model.file must be a non-empty string")
        path = Path(filename)
        if base_dir is not None:
            path = (base_dir / path).resolve()
        result["file"] = str(path)
    if "locked_joints" in config:
        locks = config["locked_joints"]
        if not isinstance(locks, Mapping):
            raise ValueError("model.locked_joints must be a TOML table")
        normalized = {}
        for name, value in locks.items():
            if not isinstance(name, str) or not name.strip() or isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"model.locked_joints.{name} must be a numeric joint position")
            position = float(value)
            if not math.isfinite(position):
                raise ValueError(f"model.locked_joints.{name} must be finite")
            normalized[name] = position
        result["locked_joints"] = normalized
    return result
