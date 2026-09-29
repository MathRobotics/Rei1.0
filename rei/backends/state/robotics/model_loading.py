"""Instantiate optional robotics backends from validated model metadata."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ....core.model_config import normalize_model_config
from .urdf import lock_urdf_joints


def _model_source(dsl: Mapping[str, Any], *, extensions: tuple[str, ...]) -> tuple[Path, dict[str, float]]:
    config = normalize_model_config(dsl.get("model", {}))
    if "file" not in config:
        raise ValueError("Provide model= explicitly or specify [model].file in the problem TOML.")
    path = Path(config["file"])
    if path.suffix.lower() not in extensions:
        raise ValueError(f"Unsupported model file format {path.suffix!r}; expected {extensions}.")
    if not path.is_file():
        raise FileNotFoundError(f"Robot model file not found: {path}")
    locks = config.get("locked_joints", {})
    if locks and path.suffix.lower() != ".urdf":
        raise ValueError("model.locked_joints requires a URDF model file.")
    return path, locks


def load_kots_model(dsl: Mapping[str, Any], *, order: int = 1, backend: str | None = None) -> Any:
    path, locks = _model_source(dsl, extensions=(".urdf", ".json"))
    from robokots.kots import Kots

    if path.suffix.lower() == ".json":
        return Kots.from_json_file(str(path), order=order, backend=backend)
    if not locks:
        return Kots.from_urdf_file(str(path), order=order, backend=backend)
    from robokots.urdf_io import urdf_xml_to_model_data

    reduced = lock_urdf_joints(path.read_text(encoding="utf-8"), locks)
    return Kots.from_json_data(urdf_xml_to_model_data(reduced.xml), order=order, backend=backend)


def load_pinocchio_model(dsl: Mapping[str, Any]) -> Any:
    path, locks = _model_source(dsl, extensions=(".urdf",))
    import pinocchio as pin

    if not locks:
        return pin.buildModelFromUrdf(str(path))
    reduced = lock_urdf_joints(path.read_text(encoding="utf-8"), locks)
    return pin.buildModelFromXML(reduced.xml)
