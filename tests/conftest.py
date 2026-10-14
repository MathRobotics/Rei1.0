from __future__ import annotations

from contextlib import contextmanager
import importlib
import sys
import types

import pytest


@pytest.fixture
def isolated_backend_imports():
    """Import backend modules against temporary dependencies, then restore caches."""

    @contextmanager
    def isolated_imports(module_names, *, stubs):
        names = tuple(dict.fromkeys((*stubs, *module_names)))
        missing = object()
        original_modules = {name: sys.modules.get(name, missing) for name in names}
        original_attributes = {}
        for name in names:
            parent_name, _, attribute = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            original_attributes[name] = (
                vars(parent).get(attribute, missing) if parent is not None else missing
            )

        try:
            for name in reversed(names):
                sys.modules.pop(name, None)
                parent_name, _, attribute = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    vars(parent).pop(attribute, None)
            sys.modules.update(stubs)
            yield {name: importlib.import_module(name) for name in module_names}
        finally:
            for name in names:
                original = original_modules[name]
                if original is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original
            for name in names:
                parent_name, _, attribute = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    original = original_attributes[name]
                    if original is missing:
                        vars(parent).pop(attribute, None)
                    else:
                        setattr(parent, attribute, original)

    return isolated_imports


@pytest.fixture
def mock_kots_modules(isolated_backend_imports):
    """Keep fake-model tests independent of installed RoboKots and test order."""
    robokots = types.ModuleType("robokots")
    core = types.ModuleType("robokots.core")
    state = types.ModuleType("robokots.core.state")

    class StateType:
        def __init__(self, owner_type, owner_name, field, frame):
            self.owner_type = owner_type
            self.owner_name = owner_name
            self.field = field
            self.frame = frame

    state.StateType = StateType
    core.state = state
    robokots.core = core
    module_names = (
        "rei.backends.state.robotics.kots_api",
        "rei.backends.state.robotics.kots_adapter",
        "rei.backends.state.robotics.kots",
        "rei.optimize_backends.kots",
    )
    with isolated_backend_imports(
        module_names,
        stubs={"robokots": robokots, "robokots.core": core, "robokots.core.state": state},
    ) as modules:
        yield modules
