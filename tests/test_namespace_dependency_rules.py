"""Keep domain implementation dependencies out of reusable lower layers."""
import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest


@pytest.mark.parametrize("layer,forbidden", [
    ("core", {"backends", "optimize", "optimize_backends", "problem", "equations", "flow"}),
    ("backends", {"optimize", "optimize_backends", "problem", "equations", "flow"}),
    ("optimize", {"backends", "optimize_backends"}),
    ("equations", {"backends", "optimize_backends"}),
])
def test_layer_import_direction(layer, forbidden):
    root = Path(__file__).resolve().parents[1]
    violations = []
    for path in (root / "rei" / layer).rglob("*.py"):
        module = ".".join(path.relative_to(root).with_suffix("").parts).removesuffix(".__init__")
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [resolve_name("." * node.level + (node.module or ""), package)
                           if node.level else node.module or ""]
            else:
                continue
            for target in imports:
                parts = target.split(".")
                if len(parts) > 1 and parts[0] == "rei" and parts[1] in forbidden:
                    violations.append(f"{path.relative_to(root)}:{node.lineno}: {target}")
    assert not violations, "\n".join(violations)
