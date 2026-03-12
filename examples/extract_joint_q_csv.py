from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Sequence

_JOINT_Q_PATTERN = re.compile(r"^joint_q\[(\d+)\]$")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract joint_q[n] columns from a log CSV into a plain numeric CSV.",
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input log CSV path, for example examples/logs/11_forward_then_inverse_ioc_*.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path. Defaults to <input_stem>_joint_q.csv next to the input file.",
    )
    return parser.parse_args(argv)


def _default_output_path(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_joint_q.csv")


def _find_joint_q_columns(fieldnames: Sequence[str] | None) -> list[str]:
    if not fieldnames:
        raise ValueError("Input CSV has no header.")

    columns: list[tuple[int, str]] = []
    for name in fieldnames:
        match = _JOINT_Q_PATTERN.fullmatch(name)
        if match is None:
            continue
        columns.append((int(match.group(1)), name))

    if not columns:
        raise ValueError("No joint_q[n] columns found in input CSV.")

    columns.sort(key=lambda item: item[0])
    return [name for _, name in columns]


def extract_joint_q_csv(input_csv: Path, output_csv: Path) -> tuple[int, int]:
    with input_csv.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        columns = _find_joint_q_columns(reader.fieldnames)

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.writer(dst, lineterminator="\n")
            row_count = 0

            for row in reader:
                if row is None:
                    continue

                values: list[str] = []
                for column in columns:
                    raw = row.get(column)
                    if raw is None or raw.strip() == "":
                        raise ValueError(
                            f"Missing value in {column!r} at data row {row_count + 1}."
                        )
                    try:
                        value = float(raw)
                    except ValueError as exc:
                        raise ValueError(
                            f"Non-numeric value {raw!r} in {column!r} at data row {row_count + 1}."
                        ) from exc
                    values.append(f"{value:.6f}")

                writer.writerow(values)
                row_count += 1

    return row_count, len(columns)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_csv = args.input_csv.resolve()
    output_csv = (args.output or _default_output_path(input_csv)).resolve()

    if not input_csv.is_file():
        print(f"error: input CSV not found: {input_csv}", file=sys.stderr)
        return 1

    try:
        row_count, joint_count = extract_joint_q_csv(input_csv, output_csv)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"wrote {row_count} rows with {joint_count} joint_q columns to {output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
