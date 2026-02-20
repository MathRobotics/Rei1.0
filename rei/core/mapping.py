from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def mapping_as_dict(mapping: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    if isinstance(mapping, dict):
        return mapping
    try:
        return dict(mapping)
    except Exception as e:
        raise TypeError(f"{where} must be a mapping.") from e


__all__ = [
    "mapping_as_dict",
]
