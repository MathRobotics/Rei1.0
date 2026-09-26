"""JSON-safe solver events and a compact, human-readable history view."""
from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class SolverHistoryRecorder:
    """Keep events in memory and/or append complete JSONL records as they occur.

    Existing files are never overwritten. Each write closes the file, including
    when evaluation subsequently fails, so completed events remain readable.
    """

    def __init__(self, *, enabled: bool, path: str | Path | None = None,
                 line_search_path: str | Path | None = None, verbose: bool = False,
                 solver: str = "gauss_newton", trial_kind: str = "line_search"):
        self.solver = solver
        self.trial_kind = trial_kind
        self.enabled = enabled
        self.verbose = verbose
        self._last_header_block: int | None = None
        self.path = None if path is None else Path(path)
        self.line_search_path = (
            Path(line_search_path) if line_search_path is not None else
            self.path.with_name(self.path.stem + (
                ".line_search.jsonl" if trial_kind == "line_search" else f".{trial_kind}_trials.jsonl"
            )) if self.path is not None else None
        )
        self.events: list[dict[str, Any]] = []
        self.line_search_events: list[dict[str, Any]] = []
        self._objective: float | None = None
        self.started = perf_counter()
        paths = [p for p in (self.path, self.line_search_path) if p is not None]
        if len({p.resolve() for p in paths}) != len(paths):
            raise ValueError("state history and trial history must use different files")
        for path in paths:
            if path.exists():
                raise FileExistsError(f"History file already exists: {path}")
            if not path.parent.is_dir():
                raise FileNotFoundError(f"History directory does not exist: {path.parent}")
        for path in paths:
            with path.open("x", encoding="utf-8"):
                pass

    @property
    def active(self) -> bool:
        return self.verbose or self.enabled or self.path is not None or self.line_search_path is not None

    def emit(self, event: str, iteration: int, **fields: Any) -> None:
        if not self.active:
            return
        if "objective" in fields:
            objective = fields["objective"]
            base = fields.get("base_objective", self._objective)
            fields["delta_objective"] = (
                0.0 if event == "initial" else
                objective - base if objective is not None and base is not None else None
            )
            if event in {"initial", "iteration_end"}:
                self._objective = objective
        row = _json_safe({
            "schema_version": 2,
            "solver": self.solver,
            "event": event,
            "iteration": int(iteration),
            "elapsed_seconds": perf_counter() - self.started,
            **fields,
        })
        is_search = event.startswith(self.trial_kind + "_")
        path = self.line_search_path if is_search else self.path
        if path is not None:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if self.enabled:
            (self.line_search_events if is_search else self.events).append(row)
        if self.verbose and event in {"initial", "iteration_end", "iteration_failed", "final"}:
            lines = format_solver_history([row]).splitlines()
            header_block = int(iteration) // 20
            if self._last_header_block is None or header_block > self._last_header_block:
                print(lines[0], flush=True)
                self._last_header_block = header_block
            print(lines[1], flush=True)


class BackendProgress:
    """Report backend iterates through the same events as built-in solvers."""

    def __init__(self, solver: str, *, verbose: bool = False):
        self.recorder = SolverHistoryRecorder(enabled=True, verbose=verbose, solver=solver)
        self.iteration = 0

    def initial(self, objective: float, gradient: Any = None) -> None:
        self.recorder.emit("initial", 0, **self._fields(objective, gradient), step_norm=None)

    def iterate(self, objective: float, step_norm: float, gradient: Any = None) -> None:
        self.iteration += 1
        self.recorder.emit("iteration_end", self.iteration,
                           **self._fields(objective, gradient), step_norm=float(step_norm))

    def final(self, outcome: Any, gradient: Any = None) -> None:
        stats = outcome.stats
        self.recorder.emit("final", int(stats.iterations),
                           **self._fields(stats.objective, gradient),
                           step_norm=stats.step_norm, status=stats.status,
                           reason=stats.status, message=stats.message)
        outcome.history = self.recorder.events

    @staticmethod
    def _fields(objective: float | None, gradient: Any) -> dict[str, Any]:
        norm = None
        if gradient is not None:
            norm = float(np.max(np.abs(np.asarray(gradient, dtype=float)), initial=0.0))
        return {
            "objective": None if objective is None else float(objective),
            "residual_norm": None if objective is None else math.sqrt(max(0.0, float(objective))),
            "jt_r_inf_norm": norm,
        }


def format_solver_history(
    history: Iterable[dict[str, Any]], *, include_line_search: bool = True,
    include_diagnostics: bool = False,
) -> str:
    """Show progress and trials; opt in to inner-solve and retry diagnostics."""
    history = list(history)
    widths = (4, 16, 9, 9, 9, 9)

    def table_row(values: Iterable[str]) -> str:
        return " ".join(
            value.ljust(width) if index == 1 else value.rjust(width)
            for index, (value, width) in enumerate(zip(values, widths, strict=True))
        )

    lines = [table_row(("iter", "event", "objective", "Δobj", "|Jᵀr|inf", "step")) + " result"]

    def number(value: Any) -> str:
        return "-" if value is None else f"{value:.2e}"

    for row in history:
        event = row["event"]
        if not include_diagnostics and event not in {
            "initial", "iteration_end", "iteration_failed", "final",
        } and not event.startswith(("line_search_", "lm_", "trust_region_")):
            continue
        if not include_line_search and event.startswith(("line_search_", "lm_", "trust_region_")):
            continue
        label = event
        if event in {"line_search_trial", "lm_trial", "trust_region_trial"}:
            label = f"  trial {row['trial']}"
        result = row.get("reason") or row.get("acceptance_reason") or row.get("line_search_status") or row.get("status") or ""
        details = []
        for key, detail_name in (("damping", "damping"), ("step_scale", "scale"),
                           ("trust_radius", "radius"), ("gain_ratio", "rho")):
            if row.get(key) is not None:
                details.append(f"{detail_name}={number(row[key])}")
        trials = row.get("line_search_trials", row.get("trials"))
        if trials is not None:
            details.append(f"trials={trials}")
        if event == "iteration_retry":
            details.append(f"retry={row['retry']}")
            details.append(f"ls={row['ls_max_iters']}")
        lines.append(
            table_row((str(row["iteration"]), label,
                       number(row.get("objective")), number(row.get("delta_objective")),
                       number(row.get("jt_r_inf_norm")), number(row.get("step_norm"))))
            + " " + " ".join(part for part in (str(result), *details) if part)
        )
    return "\n".join(lines)


__all__ = ["format_solver_history"]
