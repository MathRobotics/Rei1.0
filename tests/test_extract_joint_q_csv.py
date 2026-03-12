from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "examples" / "extract_joint_q_csv.py"


def test_extract_joint_q_csv_writes_headerless_numeric_csv(tmp_path) -> None:
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    input_csv.write_text(
        "x_axis,k,x,joint_q[2],joint_q[0],joint_q[1],joint_torque[0]\n"
        "time,0,0,0.6,-0.6,0.0,1.0\n"
        "time,1,0.01,0.600008,-0.600008,0.349976,2.0\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(input_csv), "--output", str(output_csv)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output_csv.read_text(encoding="utf-8") == (
        "-0.600000,0.000000,0.600000\n"
        "-0.600008,0.349976,0.600008\n"
    )


def test_extract_joint_q_csv_rejects_missing_joint_q_columns(tmp_path) -> None:
    input_csv = tmp_path / "input.csv"
    input_csv.write_text(
        "x_axis,k,x,joint_torque[0]\n"
        "time,0,0,1.0\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(input_csv)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "No joint_q[n] columns found" in result.stderr
