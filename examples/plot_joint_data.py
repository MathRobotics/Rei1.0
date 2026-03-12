from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

_JOINT_Q_PATTERN = re.compile(r"^joint_q\[(\d+)\]$")
_JOINT_TORQUE_PATTERN = re.compile(r"^joint_torque\[(\d+)\]$")


@dataclass
class PlotData:
    x: list[float]
    x_label: str
    joint_q: dict[str, list[float]]
    joint_torque: dict[str, list[float]]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot joint_q[n] and joint_torque[n] data from a CSV file.",
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV path, for example examples/logs/11_forward_then_inverse_ioc_20260302_151742.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output image path. If omitted, the plot is shown interactively.",
    )
    parser.add_argument(
        "--ylim-q",
        dest="joint_q_ylim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Optional y-axis limits for the joint_q plot.",
    )
    parser.add_argument(
        "--ylim-torque",
        dest="joint_torque_ylim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Optional y-axis limits for the joint_torque plot.",
    )
    return parser.parse_args(argv)


def _sorted_columns(
    fieldnames: Sequence[str] | None,
    pattern: re.Pattern[str],
) -> list[str]:
    if not fieldnames:
        return []

    columns: list[tuple[int, str]] = []
    for name in fieldnames:
        match = pattern.fullmatch(name)
        if match is None:
            continue
        columns.append((int(match.group(1)), name))

    columns.sort(key=lambda item: item[0])
    return [name for _, name in columns]


def _looks_like_log_header(first_row: Sequence[str]) -> bool:
    if "x" in first_row:
        return True
    return any(
        _JOINT_Q_PATTERN.fullmatch(value) or _JOINT_TORQUE_PATTERN.fullmatch(value)
        for value in first_row
    )


def _load_log_csv(input_csv: Path) -> PlotData:
    with input_csv.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        joint_q_columns = _sorted_columns(reader.fieldnames, _JOINT_Q_PATTERN)
        joint_torque_columns = _sorted_columns(reader.fieldnames, _JOINT_TORQUE_PATTERN)

        if not joint_q_columns and not joint_torque_columns:
            raise ValueError(
                "Input CSV has no joint_q[n] or joint_torque[n] columns."
            )

        x_label = "x" if reader.fieldnames and "x" in reader.fieldnames else "sample"
        x_values: list[float] = []
        joint_q = {column: [] for column in joint_q_columns}
        joint_torque = {column: [] for column in joint_torque_columns}

        for row_index, row in enumerate(reader):
            if row is None:
                continue

            raw_x = row.get("x")
            try:
                x_value = float(raw_x) if raw_x not in (None, "") else float(row_index)
            except ValueError:
                continue

            row_values: dict[str, float] = {}
            try:
                for column in joint_q_columns:
                    raw = row.get(column)
                    if raw is None or raw.strip() == "":
                        raise ValueError
                    row_values[column] = float(raw)
                for column in joint_torque_columns:
                    raw = row.get(column)
                    if raw is None or raw.strip() == "":
                        raise ValueError
                    row_values[column] = float(raw)
            except ValueError:
                continue

            x_values.append(x_value)
            for column in joint_q_columns:
                joint_q[column].append(row_values[column])
            for column in joint_torque_columns:
                joint_torque[column].append(row_values[column])

    if not x_values:
        raise ValueError("No plottable numeric rows found in input CSV.")

    return PlotData(
        x=x_values,
        x_label=x_label,
        joint_q=joint_q,
        joint_torque=joint_torque,
    )


def _load_plain_joint_q_csv(input_csv: Path) -> PlotData:
    joint_q_rows: list[list[float]] = []

    with input_csv.open("r", encoding="utf-8", newline="") as src:
        reader = csv.reader(src)
        for row in reader:
            if not row:
                continue
            try:
                joint_q_rows.append([float(value) for value in row])
            except ValueError as exc:
                raise ValueError(
                    "Plain joint_q CSV must contain only numeric values."
                ) from exc

    if not joint_q_rows:
        raise ValueError("Input CSV is empty.")

    joint_count = len(joint_q_rows[0])
    if joint_count == 0:
        raise ValueError("Input CSV has no columns.")
    if any(len(row) != joint_count for row in joint_q_rows):
        raise ValueError("Plain joint_q CSV has inconsistent column counts.")

    joint_q = {
        f"joint_q[{index}]": [row[index] for row in joint_q_rows]
        for index in range(joint_count)
    }
    x_values = [float(index) for index in range(len(joint_q_rows))]

    return PlotData(
        x=x_values,
        x_label="sample",
        joint_q=joint_q,
        joint_torque={},
    )


def load_plot_data(input_csv: Path) -> PlotData:
    with input_csv.open("r", encoding="utf-8", newline="") as src:
        reader = csv.reader(src)
        first_row = next(reader, None)

    if first_row is None:
        raise ValueError("Input CSV is empty.")

    if _looks_like_log_header(first_row):
        return _load_log_csv(input_csv)
    return _load_plain_joint_q_csv(input_csv)


def _normalize_ylim(
    option_name: str,
    values: Sequence[float] | None,
) -> tuple[float, float] | None:
    if values is None:
        return None

    lower, upper = float(values[0]), float(values[1])
    if lower >= upper:
        raise ValueError(f"{option_name} requires MIN < MAX.")
    return (lower, upper)


def plot_joint_data(
    data: PlotData,
    title: str | None = None,
    joint_q_ylim: tuple[float, float] | None = None,
    joint_torque_ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    plot_groups: list[
        tuple[str, str, dict[str, list[float]], tuple[float, float] | None]
    ] = []
    if data.joint_q:
        plot_groups.append(
            ("Joint Angles", "Joint angle [rad]", data.joint_q, joint_q_ylim)
        )
    if data.joint_torque:
        plot_groups.append(
            ("Joint Torques", "Joint torque", data.joint_torque, joint_torque_ylim)
        )

    if not plot_groups:
        raise ValueError("No plot data found.")

    fig, axes = plt.subplots(
        len(plot_groups),
        1,
        figsize=(10, 4 * len(plot_groups)),
        sharex=True,
    )
    axes_list = [axes] if len(plot_groups) == 1 else list(axes)

    for axis, (group_title, y_label, series, ylim) in zip(axes_list, plot_groups):
        for label, values in series.items():
            axis.plot(data.x, values, label=label)
        axis.set_ylabel(y_label)
        axis.set_title(group_title)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(True)
        axis.legend()

    axes_list[-1].set_xlabel(data.x_label)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_csv = args.input_csv.resolve()
    output_path = args.output.resolve() if args.output else None

    if not input_csv.is_file():
        print(f"error: input CSV not found: {input_csv}", file=sys.stderr)
        return 1

    try:
        joint_q_ylim = _normalize_ylim("--ylim-q", args.joint_q_ylim)
        joint_torque_ylim = _normalize_ylim("--ylim-torque", args.joint_torque_ylim)
        data = load_plot_data(input_csv)
        fig = plot_joint_data(
            data,
            title=input_csv.name,
            joint_q_ylim=joint_q_ylim,
            joint_torque_ylim=joint_torque_ylim,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"wrote plot to {output_path}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
