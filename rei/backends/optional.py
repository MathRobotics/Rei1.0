from __future__ import annotations

import importlib
from types import ModuleType


def import_optional_backend(module_name: str, *, backend_name: str, install_hint: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"`{backend_name}` backend support requires optional dependency `{module_name}`. "
            f"Install it with `{install_hint}` and retry."
        ) from exc


def require_module_attrs(
    module: ModuleType,
    attrs: tuple[str, ...],
    *,
    backend_name: str,
    install_hint: str,
    extra_hint: str | None = None,
) -> None:
    missing = [name for name in attrs if not hasattr(module, name)]
    if not missing:
        return

    hint = "" if extra_hint is None else f" {extra_hint}"
    raise ImportError(
        f"`{backend_name}` backend support found module `{module.__name__}`, but it is missing "
        f"required APIs: {', '.join(missing)}.{hint} "
        f"Install the expected package with `{install_hint}` and retry."
    )


__all__ = ["import_optional_backend", "require_module_attrs"]
