"""Integration CI must fail on both module-level and runtime skips."""
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("source,strict,expected", [
    ("def test_skip():\n import pytest; pytest.skip('unavailable')\n", True, 1),
    ("import pytest\npytest.skip('unavailable', allow_module_level=True)\n", True, 5),
    ("def test_skip():\n import pytest; pytest.skip('unavailable')\n", False, 0),
    ("def test_pass():\n assert True\n", True, 0),
])
def test_skip_exit_status(tmp_path, source, strict, expected):
    (tmp_path / "conftest.py").write_text(Path(__file__).with_name("conftest.py").read_text())
    test = tmp_path / "test_sample.py"
    test.write_text(source)
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n")
    command = [sys.executable, "-m", "pytest", "-q", "-c", str(config), str(test)]
    if strict:
        command.append("--fail-on-skip")
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == expected, result.stdout + result.stderr
    if strict and "skip" in source:
        assert "unexpected skip(s)" in result.stdout
