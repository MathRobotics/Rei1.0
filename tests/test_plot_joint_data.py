from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "examples" / "plot_joint_data.py"


def _load_plot_module():
    spec = importlib.util.spec_from_file_location("plot_joint_data_example", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_plot_mod = _load_plot_module()


def _run_plot_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_plot_joint_data_writes_image_for_log_csv(tmp_path) -> None:
    input_csv = tmp_path / "input.csv"
    output_png = tmp_path / "plot.png"
    input_csv.write_text(
        "x_axis,k,x,joint_q[1],joint_torque[0],joint_q[0],joint_torque[1]\n"
        "time,0,0.0,1.5,-2.0,0.5,3.0\n"
        "time,1,0.1,1.4,-1.5,0.6,2.5\n",
        encoding="utf-8",
    )

    result = _run_plot_script(
        str(input_csv),
        "--output",
        str(output_png),
        "--ylim-q",
        "-1.0",
        "2.0",
        "--ylim-torque",
        "-3.0",
        "4.0",
    )

    assert result.returncode == 0, result.stderr
    assert output_png.is_file()
    assert output_png.stat().st_size > 0


def test_plot_joint_data_writes_image_for_plain_joint_q_csv(tmp_path) -> None:
    input_csv = tmp_path / "input.csv"
    output_png = tmp_path / "plot.png"
    input_csv.write_text(
        "0.1,0.2,0.3\n"
        "0.4,0.5,0.6\n",
        encoding="utf-8",
    )

    result = _run_plot_script(str(input_csv), "--output", str(output_png))

    assert result.returncode == 0, result.stderr
    assert output_png.is_file()
    assert output_png.stat().st_size > 0


def test_plot_joint_data_writes_image_for_headered_joint_q_csv_without_x(tmp_path) -> None:
    input_csv = tmp_path / "input.csv"
    output_png = tmp_path / "plot.png"
    input_csv.write_text(
        "joint_q[1],joint_q[0]\n"
        "1.5,0.5\n"
        "1.4,0.6\n",
        encoding="utf-8",
    )

    result = _run_plot_script(str(input_csv), "--output", str(output_png))

    assert result.returncode == 0, result.stderr
    assert output_png.is_file()
    assert output_png.stat().st_size > 0


def test_plot_joint_data_applies_requested_y_limits() -> None:
    data = _plot_mod.PlotData(
        x=[0.0, 1.0],
        x_label="sample",
        joint_q={"joint_q[0]": [0.2, 0.4]},
        joint_torque={"joint_torque[0]": [-1.0, 1.5]},
    )

    fig = _plot_mod.plot_joint_data(
        data,
        joint_q_ylim=(-0.5, 0.5),
        joint_torque_ylim=(-2.0, 2.0),
    )

    assert fig.axes[0].get_ylim() == (-0.5, 0.5)
    assert fig.axes[1].get_ylim() == (-2.0, 2.0)
    plt.close(fig)


def test_plot_joint_data_rejects_csv_without_plot_columns(tmp_path) -> None:
    input_csv = tmp_path / "input.csv"
    input_csv.write_text(
        "x_axis,k,x\n"
        "time,0,0.0\n",
        encoding="utf-8",
    )

    result = _run_plot_script(str(input_csv))

    assert result.returncode == 1
    assert "no joint_q[n] or joint_torque[n] columns" in result.stderr.lower()


def test_plot_joint_data_rejects_invalid_ylim_range(tmp_path) -> None:
    input_csv = tmp_path / "input.csv"
    input_csv.write_text(
        "0.1,0.2,0.3\n"
        "0.4,0.5,0.6\n",
        encoding="utf-8",
    )

    result = _run_plot_script(str(input_csv), "--ylim-q", "1.0", "1.0")

    assert result.returncode == 1
    assert "--ylim-q requires min < max" in result.stderr.lower()
