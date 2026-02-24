from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Callable

import numpy as np

from ..core.outcome import SolveOutcome
from .report import format_timing_report

IterRow = tuple[int, float, float]
IterCallback = Callable[[int, float, float], None]


def build_solver_iter_logger(
    solver: str,
    options: Mapping[str, Any] | None = None,
    *,
    verbose_solver: str = "gauss_newton",
    verbose: bool | None = None,
    verbose_every: int | None = None,
    strip_verbose_options: bool = True,
    print_prefix: str | None = None,
) -> tuple[dict[str, Any], IterCallback, list[IterRow]]:
    """Build an `on_iter` callback and iteration history container.

    Returns:
      - options_local: options dict with local verbose keys removed when configured
      - on_iter: callback compatible with `solve(..., on_iter=...)`
      - history: mutable list of `(iter, rnorm, dxnorm)` tuples
    """

    opts = {} if options is None else dict(options)
    solver_key = str(solver)
    verbose_solver_key = str(verbose_solver)
    history: list[IterRow] = []

    verbose_enabled = False
    every_i = 1
    if solver_key == verbose_solver_key:
        if verbose is None:
            verbose_enabled = bool(opts.pop("verbose", False)) if strip_verbose_options else bool(opts.get("verbose", False))
        else:
            verbose_enabled = bool(verbose)

        if verbose_every is None:
            every_raw = opts.pop("verbose_every", 1) if strip_verbose_options else opts.get("verbose_every", 1)
        else:
            every_raw = verbose_every
        every_i = int(every_raw)
        if every_i <= 0:
            raise ValueError(f"build_solver_iter_logger: verbose_every must be > 0, got {every_raw!r}.")

        if verbose_enabled:
            prefix = solver_key if print_prefix is None else str(print_prefix)
            print(f"[{prefix}] {solver_key} verbose enabled (every={every_i})")

    def _on_iter(k: int, rnorm: float, dxnorm: float) -> None:
        k_i = int(k)
        r_f = float(rnorm)
        dx_f = float(dxnorm)
        history.append((k_i, r_f, dx_f))

        if solver_key == verbose_solver_key and verbose_enabled and k_i % every_i == 0:
            prefix = solver_key if print_prefix is None else str(print_prefix)
            print(f"[{prefix}:{solver_key}] iter={k_i:04d} rnorm={r_f:.3e} dxnorm={dx_f:.3e}")

    return opts, _on_iter, history


def compress_iter_history(rows: Iterable[IterRow]) -> list[IterRow]:
    """Deduplicate callbacks by iteration index while preserving first-seen order."""

    latest: dict[int, IterRow] = {}
    order: list[int] = []
    for k, rnorm, dxnorm in rows:
        k_i = int(k)
        if k_i not in latest:
            order.append(k_i)
        latest[k_i] = (k_i, float(rnorm), float(dxnorm))
    return [latest[k] for k in order]


def format_numeric_array(
    values: np.ndarray | Sequence[float],
    *,
    precision: int = 6,
) -> str:
    x_vec = np.asarray(values, dtype=float).reshape(-1)
    return np.array2string(x_vec, precision=int(precision), separator=", ", suppress_small=True)


def build_timestamped_log_path(
    log_dir: str | Path,
    *,
    prefix: str,
    timestamp: datetime | None = None,
    suffix: str = "txt",
) -> Path:
    now = datetime.now() if timestamp is None else timestamp
    suffix_clean = str(suffix).strip().lstrip(".")
    if suffix_clean == "":
        raise ValueError("build_timestamped_log_path: suffix must be non-empty.")
    ts = now.strftime("%Y%m%d_%H%M%S")
    return Path(log_dir) / f"{str(prefix)}_{ts}.{suffix_clean}"


def write_text_log(
    path: str | Path,
    text: str,
    *,
    ensure_parent: bool = True,
    ensure_trailing_newline: bool = True,
) -> None:
    p = Path(path)
    if ensure_parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    out = str(text)
    if ensure_trailing_newline and not out.endswith("\n"):
        out += "\n"
    p.write_text(out, encoding="utf-8")


def _append_pformat(lines: list[str], mapping: Mapping[str, Any]) -> None:
    lines.extend(pformat(dict(mapping), sort_dicts=True).splitlines())


def _append_section(lines: list[str], name: str, body: Any) -> None:
    lines.append("")
    lines.append(f"[{str(name)}]")
    if isinstance(body, Mapping):
        _append_pformat(lines, body)
        return
    if isinstance(body, str):
        lines.extend(body.splitlines())
        return
    if isinstance(body, Iterable):
        lines.extend(str(x) for x in body)
        return
    lines.append(str(body))


def format_solver_text_log(
    *,
    title: str,
    solver: str,
    outcome: SolveOutcome,
    requested_options: Mapping[str, Any] | None = None,
    solve_options: Mapping[str, Any] | None = None,
    iter_history: Iterable[IterRow] | None = None,
    header: Mapping[str, Any] | None = None,
    timestamp: datetime | None = None,
    include_timing: bool = True,
    timing_title: str = "timing",
    deduplicate_iter: bool = True,
    extra_sections: Iterable[tuple[str, Any]] | None = None,
) -> str:
    """Format a plain-text run log around a `SolveOutcome`."""

    now = datetime.now().astimezone() if timestamp is None else timestamp
    stats = outcome.stats

    lines: list[str] = []
    lines.append(f"=== {str(title)} ===")
    lines.append(f"timestamp={now.isoformat(timespec='seconds')}")

    if header is not None:
        for k, v in header.items():
            lines.append(f"{str(k)}={v}")

    lines.append("")
    lines.append("[solve.settings]")
    lines.append(f"solver={str(solver)}")
    lines.append("requested_options:")
    _append_pformat(lines, {} if requested_options is None else requested_options)
    lines.append("solve_options:")
    _append_pformat(lines, {} if solve_options is None else solve_options)

    lines.append("")
    lines.append("[solve.result]")
    lines.append(
        f"status={stats.status} converged={stats.converged} iters={stats.iterations} "
        f"cost0={float(stats.initial_objective or 0.0):.12e} "
        f"cost={float(stats.objective or 0.0):.12e} "
        f"rnorm={float(stats.residual_norm or 0.0):.12e} "
        f"dxnorm={float(stats.step_norm or 0.0):.12e}"
    )
    if str(stats.message):
        lines.append(f"message={stats.message}")
    lines.append("meta:")
    _append_pformat(lines, dict(outcome.meta))

    if include_timing:
        lines.append("")
        lines.extend(format_timing_report(outcome.timing, title=str(timing_title)).splitlines())

    if iter_history is not None:
        rows = list(iter_history)
        if deduplicate_iter:
            rows = compress_iter_history(rows)
        lines.append("")
        lines.append("[solve.iter_log]")
        lines.append(f"n_rows={len(rows)}")
        if len(rows) == 0:
            lines.append("(empty)")
        else:
            lines.append("iter,rnorm,dxnorm")
            for k, rnorm, dxnorm in rows:
                lines.append(f"{int(k):04d},{float(rnorm):.12e},{float(dxnorm):.12e}")

    if extra_sections is not None:
        for name, body in extra_sections:
            _append_section(lines, str(name), body)

    return "\n".join(lines)


__all__ = [
    "IterRow",
    "IterCallback",
    "build_solver_iter_logger",
    "compress_iter_history",
    "format_numeric_array",
    "build_timestamped_log_path",
    "write_text_log",
    "format_solver_text_log",
]
