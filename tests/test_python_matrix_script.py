"""Check matrix-runner failure propagation without downloading interpreters."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/test-python.sh"


@pytest.fixture
def runner(tmp_path):
    audit = tmp_path / "calls.jsonl"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = tmp_path / "python-stub"
    python_stub.write_text(f"#!{sys.executable}\n" + textwrap.dedent(r'''
        import json, os, sys
        from pathlib import Path
        case = Path(sys.argv[0]).resolve().parent.parent.name
        with open(os.environ['MATRIX_AUDIT'], 'a') as file:
            file.write(json.dumps({'case': case, 'python': sys.argv[1:]}) + '\n')
        if sys.argv[1:3] == ['-m', 'pytest']:
            if case.startswith('3.12-') and os.environ.get('MATRIX_FAIL') == 'pytest':
                sys.exit(9)
        print('stub command completed')
    '''))
    uv = bin_dir / "uv"
    uv.write_text(f"#!{sys.executable}\n" + textwrap.dedent(r'''
        import json, os, shutil, sys
        from pathlib import Path
        args = sys.argv[1:]
        if args[0] == 'sync':
            environment = Path(os.environ['UV_PROJECT_ENVIRONMENT'])
        else:
            environment = Path(args[args.index('--python') + 1]).parent.parent
        case = environment.name
        with open(os.environ['MATRIX_AUDIT'], 'a') as file:
            file.write(json.dumps({'case': case, 'uv': args}) + '\n')
        phase = 'sync' if args[0] == 'sync' else args[1]
        if case.startswith('3.12-') and os.environ.get('MATRIX_FAIL') == phase:
            sys.exit(7)
        if args[0] == 'sync':
            (environment / 'bin').mkdir(parents=True, exist_ok=True)
            shutil.copy2(os.environ['MATRIX_PYTHON_STUB'], environment / 'bin/python')
    '''))
    python_stub.chmod(0o755)
    uv.chmod(0o755)

    def run(arguments, fail=""):
        env = os.environ | {
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "MATRIX_AUDIT": str(audit), "MATRIX_FAIL": fail,
            "MATRIX_PYTHON_STUB": str(python_stub),
        }
        completed = subprocess.run([
            "bash", str(SCRIPT), "--env-dir", str(tmp_path / "env space"),
            "--log-dir", str(tmp_path / "log space"), *arguments,
        ], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
        calls = [json.loads(line) for line in audit.read_text().splitlines()] if audit.exists() else []
        summaries = list((tmp_path / "log space").glob("*/summary.tsv"))
        return completed, calls, summaries
    return run


@pytest.mark.parametrize("phase", ["sync", "check", "pytest"])
def test_matrix_continues_after_failed_version(runner, phase):
    result, calls, summaries = runner(["3.11", "3.12", "3.13"], fail=phase)
    assert result.returncode == 1, result.stdout + result.stderr
    summary = summaries[0].read_text()
    assert "3.11\tcore\tPASS\t0" in summary
    assert "3.12\tcore\tFAIL" in summary
    assert "3.13\tcore\tPASS\t0" in summary
    assert any(call.get("python", [])[:2] == ["-m", "pytest"]
               and call["case"] == "3.13-core" for call in calls)
    for call in calls:
        if call.get("uv", [])[:1] == ["sync"]:
            assert "--locked" in call["uv"]


def test_robotics_runs_both_portable_selections(runner):
    result, calls, summaries = runner(["--suite", "robotics", "3.13"])
    assert result.returncode == 0, result.stdout + result.stderr
    tests = [call["python"] for call in calls if call.get("python", [])[:2] == ["-m", "pytest"]]
    assert len(tests) == 2
    assert all("--fail-on-skip" in args for args in tests)
    assert "not rust" in tests[0] and "numpy" in tests[1]
    assert "3.13\trobotics\tPASS\t0" in summaries[0].read_text()


def test_unsupported_python_rejected_before_uv(runner):
    result, calls, summaries = runner(["3.10"])
    assert result.returncode == 2
    assert "requires Python >=3.11" in result.stderr
    assert calls == [] and summaries == []
