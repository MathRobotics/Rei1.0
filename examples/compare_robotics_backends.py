from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rei import load_problem_spec_toml, solve
from rei.optimize.reductions import build_nullspace_equality_reduction
from rei.optimize_backends.trajectory_ioc import compile_trajectory_ioc_problem, estimate_ioc_weights


_EXAMPLES_DIR = Path(__file__).resolve().parent
_DEFAULT_PIN_SPEC = _EXAMPLES_DIR / "spec" / "pinocchio_traj_dynamics.toml"
_DEFAULT_KOTS_SPEC = _EXAMPLES_DIR / "spec" / "robokots_traj_dynamics_d12.toml"
_DEFAULT_PIN_MODEL = _EXAMPLES_DIR / "models" / "planar2.urdf"
_DEFAULT_KOTS_MODEL = _EXAMPLES_DIR / "models" / "planar2.json"


@dataclass(frozen=True)
class BackendConfig:
    backend: str
    spec_path: Path
    model_path: Path
    order: int


@dataclass
class BackendResult:
    backend: str
    ok: bool
    error: str | None = None
    compile_s: float | None = None
    reduction_s: float | None = None
    solve_s: float | None = None
    ioc_s: float | None = None
    steps: int | None = None
    p_dim: int | None = None
    dynamics_fields: tuple[str, ...] = ()
    reduction_rank: int | None = None
    reduced_dim: int | None = None
    solve_status: str | None = None
    solve_converged: bool | None = None
    solve_iters: int | None = None
    solve_cost: float | None = None
    ioc_terms: int | None = None
    ioc_active_terms: int | None = None
    ioc_norm: float | None = None
    solve_spans: tuple[tuple[str, float, int], ...] = ()


def _load_backend_model(config: BackendConfig) -> tuple[Any, Any]:
    if config.backend == "pinocchio":
        try:
            import pinocchio as pin
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Pinocchio is not installed. Install it with `uv sync --group pinocchio`."
            ) from e
        model = pin.buildModelFromUrdf(str(config.model_path))
        return model, model.createData()

    if config.backend == "robokots":
        try:
            from robokots.kots import Kots
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("RoboKots is not installed. Install it with `uv sync --group kots`.") from e
        model = Kots.from_json_file(str(config.model_path), order=int(config.order))
        return model, model.state_dict_

    raise ValueError(f"unsupported backend: {config.backend!r}")


def _filter_terms(problem: dict[str, Any], selected: tuple[str, ...]) -> dict[str, Any]:
    if not selected:
        return problem

    selected_set = {name.strip() for name in selected if name.strip()}
    if not selected_set:
        return problem

    terms = []
    for term in problem.get("terms", []):
        expr = term.get("expr", {}) if isinstance(term, dict) else {}
        names = {
            str(term.get("name", "")),
            str(expr.get("name", "")) if isinstance(expr, dict) else "",
        }
        if names & selected_set:
            terms.append(term)
    out = dict(problem)
    out["terms"] = terms
    return out


def _run_backend(
    config: BackendConfig,
    *,
    term_filter: tuple[str, ...],
    run_solve: bool,
    run_ioc: bool,
    solver_max_iters: int,
) -> BackendResult:
    result = BackendResult(backend=config.backend, ok=False)
    try:
        if not config.spec_path.is_file():
            raise FileNotFoundError(f"spec file not found: {config.spec_path}")
        if not config.model_path.is_file():
            raise FileNotFoundError(f"model file not found: {config.model_path}")

        problem = load_problem_spec_toml(config.spec_path)
        problem = _filter_terms(problem, term_filter)
        model, data = _load_backend_model(config)

        t0 = time.perf_counter()
        compiled = compile_trajectory_ioc_problem(
            problem,
            backend=config.backend,
            model=model,
            data=data,
        )
        result.compile_s = time.perf_counter() - t0
        result.steps = int(compiled.trajectory_map.steps)
        result.p_dim = int(compiled.trajectory_map.p_dim)
        result.dynamics_fields = tuple(getattr(compiled.compiled, "dynamics_fields", ()))

        runtime = compiled.runtime
        if run_solve:
            t0 = time.perf_counter()
            reduction = build_nullspace_equality_reduction(
                runtime,
                eq_selector_attr="enforce",
                eq_selector_value="nullspace",
            )
            result.reduction_s = time.perf_counter() - t0
            result.reduction_rank = int(reduction.rank)
            result.reduced_dim = int(reduction.runtime.pack.n_total)

            t0 = time.perf_counter()
            solve_out = solve(
                reduction.runtime,
                solver="gauss_newton",
                options={"max_iters": int(solver_max_iters), "tol_dx": 1e-8},
            )
            result.solve_s = time.perf_counter() - t0
            reduction.runtime.update_state_if_needed()
            result.solve_status = str(solve_out.stats.status)
            result.solve_converged = bool(solve_out.stats.converged)
            result.solve_iters = int(solve_out.stats.iterations)
            result.solve_cost = None if solve_out.stats.objective is None else float(solve_out.stats.objective)
            timing = getattr(solve_out, "timing", None)
            spans = getattr(timing, "spans", ()) if timing is not None else ()
            result.solve_spans = tuple(
                (str(span.name), float(span.seconds), int(span.count))
                for span in spans
            )

        if run_ioc:
            t0 = time.perf_counter()
            ioc = estimate_ioc_weights(compiled)
            result.ioc_s = time.perf_counter() - t0
            result.ioc_terms = len(ioc.get("terms", []))
            result.ioc_active_terms = len(ioc.get("active_terms", []))
            stationarity = ioc.get("stationarity", {})
            norm = stationarity.get("ikkt_residual_norm", None) if isinstance(stationarity, dict) else None
            result.ioc_norm = None if norm is None else float(norm)

        result.ok = True
        return result
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        return result


def _format_float(value: float | None, *, precision: int = 3) -> str:
    if value is None:
        return "-"
    if not np.isfinite(float(value)):
        return str(value)
    return f"{float(value):.{precision}f}"


def _print_results(results: list[BackendResult], *, details: bool = False) -> None:
    headers = [
        "backend",
        "ok",
        "compile",
        "reduce",
        "solve",
        "ioc",
        "steps",
        "p_dim",
        "dyn",
        "rank",
        "iters",
        "cost",
        "ioc_terms",
        "ioc_norm",
    ]
    rows: list[list[str]] = []
    for r in results:
        rows.append(
            [
                r.backend,
                "yes" if r.ok else "no",
                _format_float(r.compile_s),
                _format_float(r.reduction_s),
                _format_float(r.solve_s),
                _format_float(r.ioc_s),
                "-" if r.steps is None else str(r.steps),
                "-" if r.p_dim is None else str(r.p_dim),
                ",".join(r.dynamics_fields) if r.dynamics_fields else "-",
                "-" if r.reduction_rank is None else f"{r.reduction_rank}/{r.reduced_dim}",
                "-" if r.solve_iters is None else str(r.solve_iters),
                _format_float(r.solve_cost, precision=6),
                "-" if r.ioc_terms is None else f"{r.ioc_active_terms}/{r.ioc_terms}",
                _format_float(r.ioc_norm, precision=6),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    print(line(headers))
    print(line(["-" * w for w in widths]))
    for row in rows:
        print(line(row))

    errors = [r for r in results if not r.ok]
    if errors:
        print()
        for r in errors:
            print(f"{r.backend} error: {r.error}")

    if details:
        print()
        for r in results:
            if not r.solve_spans:
                continue
            print(f"{r.backend} solve timing:")
            total = sum(seconds for _name, seconds, _count in r.solve_spans)
            for name, seconds, count in sorted(r.solve_spans, key=lambda item: item[1], reverse=True):
                share = 0.0 if total <= 0.0 else 100.0 * float(seconds) / float(total)
                print(f"  {name:<24} {seconds:9.6f}s  {share:5.1f}%  count={count}")


def _parse_terms(raw: str | None) -> tuple[str, ...]:
    if raw is None or raw.strip() == "":
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Pinocchio and RoboKots trajectory backends.")
    parser.add_argument("--backend", choices=("both", "pinocchio", "robokots"), default="both")
    parser.add_argument("--pinocchio-spec", type=Path, default=_DEFAULT_PIN_SPEC)
    parser.add_argument("--robokots-spec", type=Path, default=_DEFAULT_KOTS_SPEC)
    parser.add_argument("--pinocchio-model", type=Path, default=_DEFAULT_PIN_MODEL)
    parser.add_argument("--robokots-model", type=Path, default=_DEFAULT_KOTS_MODEL)
    parser.add_argument("--robokots-order", type=int, default=5)
    parser.add_argument(
        "--terms",
        default=None,
        help="Comma-separated term expr/name filter, e.g. qdot_traj_regularization,torque_traj_regularization.",
    )
    parser.add_argument("--skip-solve", action="store_true", help="Only compile and optionally estimate IOC weights.")
    parser.add_argument("--skip-ioc", action="store_true", help="Skip estimate_ioc_weights timing.")
    parser.add_argument("--solver-max-iters", type=int, default=500)
    parser.add_argument("--details", action="store_true", help="Print solver timing spans for each backend.")
    args = parser.parse_args()

    configs = []
    if args.backend in ("both", "pinocchio"):
        configs.append(
            BackendConfig(
                backend="pinocchio",
                spec_path=args.pinocchio_spec,
                model_path=args.pinocchio_model,
                order=0,
            )
        )
    if args.backend in ("both", "robokots"):
        configs.append(
            BackendConfig(
                backend="robokots",
                spec_path=args.robokots_spec,
                model_path=args.robokots_model,
                order=int(args.robokots_order),
            )
        )

    results = [
        _run_backend(
            config,
            term_filter=_parse_terms(args.terms),
            run_solve=not bool(args.skip_solve),
            run_ioc=not bool(args.skip_ioc),
            solver_max_iters=int(args.solver_max_iters),
        )
        for config in configs
    ]
    _print_results(results, details=bool(args.details))


if __name__ == "__main__":
    main()
