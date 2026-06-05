from __future__ import annotations

from pathlib import Path

from rei.optimize.dsl import load_problem_spec_toml


def test_problem_spec_examples_load() -> None:
    spec_dir = Path(__file__).resolve().parents[1] / "examples" / "spec"
    paths = sorted(spec_dir.glob("*.toml"))

    assert paths
    for path in paths:
        dsl = load_problem_spec_toml(path)
        assert isinstance(dsl.get("terms"), list), path.name
