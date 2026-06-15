from __future__ import annotations

import argparse
import gc
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rei import load_problem_spec_toml, solve
from rei.optimize.reductions import build_nullspace_equality_reduction
from rei.optimize_backends.kots import compile_kots_trajectory_problem
from rei.optimize_backends.pinocchio import compile_pinocchio_trajectory_problem

try:
    import pinocchio as pin
except ImportError as e:  # pragma: no cover
    raise SystemExit("This benchmark requires Pinocchio. Install with `uv sync --group pinocchio`.") from e

try:
    from robokots.kots import Kots
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This benchmark requires RoboKots. For a local checkout, run for example:\n"
        "  PYTHONPATH=/path/to/RoboKots:. python developer/benchmarks/urdf_rust_kots_vs_pinocchio.py"
    ) from e


ROOT = Path(__file__).resolve().parents[2]


def median_ms(values: list[float]) -> float:
    return 1000.0 * float(statistics.median(values))


def bench(fn: Callable[[], Any], *, repeat: int, warmup: int) -> list[float]:
    for _ in range(int(warmup)):
        fn()
    gc.collect()

    values: list[float] = []
    for _ in range(int(repeat)):
        t0 = time.perf_counter()
        fn()
        values.append(time.perf_counter() - t0)
    return values


def set_runtime_point(runtime: Any, x: np.ndarray) -> None:
    current = np.asarray(runtime.pack.get(), dtype=float).reshape(-1)
    runtime.pack.apply_dx(np.asarray(x, dtype=float).reshape(-1) - current)


def build_pinocchio(urdf_path: Path) -> tuple[Any, Any]:
    model = pin.buildModelFromUrdf(str(urdf_path))
    return model, model.createData()


def build_kots(urdf_path: Path, *, order: int) -> Any:
    return Kots.from_urdf_file(str(urdf_path), order=int(order), dim=3)


def compile_problem(backend: str, problem: dict[str, Any], urdf_path: Path, *, order: int) -> Any:
    if backend == "pinocchio":
        model, data = build_pinocchio(urdf_path)
        return compile_pinocchio_trajectory_problem(problem, model=model, data=data)
    if backend == "kots-default":
        model = build_kots(urdf_path, order=order)
        return compile_kots_trajectory_problem(problem, model=model, data=model.state_dict_, jacobian_strategy="mul")
    if backend == "kots-rust":
        model = build_kots(urdf_path, order=order)
        return compile_kots_trajectory_problem(
            problem,
            model=model,
            data=model.state_dict_,
            jacobian_strategy="mul",
            kots_backend="rust",
        )
    raise ValueError(f"unknown backend: {backend}")


def measure_backend(
    *,
    backend: str,
    problem: dict[str, Any],
    urdf_path: Path,
    order: int,
    repeat: int,
    warmup: int,
    solve_repeat: int,
) -> dict[str, Any]:
    if backend == "pinocchio":
        load_times = bench(lambda: build_pinocchio(urdf_path), repeat=repeat, warmup=warmup)
    elif backend == "kots-default":
        load_times = bench(lambda: build_kots(urdf_path, order=order), repeat=repeat, warmup=warmup)
    else:
        load_times = bench(lambda: build_kots(urdf_path, order=order), repeat=repeat, warmup=warmup)

    compile_times = bench(
        lambda: compile_problem(backend, problem, urdf_path, order=order),
        repeat=repeat,
        warmup=warmup,
    )
    compiled = compile_problem(backend, problem, urdf_path, order=order)
    runtime = compiled.runtime
    n_total = int(runtime.pack.n_total)

    rng = np.random.default_rng(0)
    xs = [0.05 * rng.standard_normal(n_total) for _ in range(repeat + warmup)]
    linearize_times: list[float] = []
    residual_shape = ()
    jacobian_shape = ()
    for i, x in enumerate(xs):
        set_runtime_point(runtime, x)
        t0 = time.perf_counter()
        residual, jacobian = runtime.linearize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            linearize_times.append(dt)
        residual_shape = residual.shape
        jacobian_shape = jacobian.shape

    solve_times: list[float] = []
    solve_statuses: list[str] = []
    solve_iters: list[int] = []
    for _ in range(int(solve_repeat)):
        compiled = compile_problem(backend, problem, urdf_path, order=order)
        reduction = build_nullspace_equality_reduction(
            compiled.runtime,
            eq_selector_attr="enforce",
            eq_selector_value="nullspace",
        )
        t0 = time.perf_counter()
        outcome = solve(
            reduction.runtime,
            solver="gauss_newton",
            options={"max_iters": 500, "tol_dx": 1e-8},
        )
        solve_times.append(time.perf_counter() - t0)
        solve_statuses.append(str(outcome.stats.status))
        solve_iters.append(int(outcome.stats.iterations))

    return {
        "load_ms": median_ms(load_times),
        "compile_ms": median_ms(compile_times),
        "linearize_ms": median_ms(linearize_times),
        "solve_ms": median_ms(solve_times),
        "n_total": n_total,
        "rows": int(residual_shape[0]),
        "jacobian_shape": jacobian_shape,
        "status": sorted(set(solve_statuses)),
        "iters": int(statistics.median(solve_iters)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Rei URDF trajectory performance: Pinocchio vs RoboKots Rust.")
    parser.add_argument("--model", default="planar2", help="Model basename under examples/models, without .urdf.")
    parser.add_argument("--spec", default="examples/spec/pinocchio_traj_dynamics.toml")
    parser.add_argument("--order", type=int, default=5, help="RoboKots model order.")
    parser.add_argument("--repeat", type=int, default=20, help="Repeat count for load/compile/linearize.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--solve-repeat", type=int, default=3)
    parser.add_argument(
        "--backend",
        action="append",
        choices=("pinocchio", "kots-default", "kots-rust"),
        help="Backend to measure. May be passed multiple times.",
    )
    args = parser.parse_args()

    urdf_path = ROOT / "examples" / "models" / f"{args.model}.urdf"
    spec_path = ROOT / args.spec
    if not urdf_path.is_file():
        raise SystemExit(f"URDF not found: {urdf_path}")
    if not spec_path.is_file():
        raise SystemExit(f"Spec not found: {spec_path}")

    problem = load_problem_spec_toml(spec_path)
    backends = args.backend or ["pinocchio", "kots-default", "kots-rust"]

    print(f"robokots={__import__('robokots').__file__}")
    print(f"pinocchio={pin.__file__}")
    print(f"model={urdf_path}")
    print(f"spec={spec_path}")
    print(f"repeat={args.repeat} warmup={args.warmup} solve_repeat={args.solve_repeat}")
    print()
    print("backend          load_ms  compile_ms  linearize_ms  solve_ms  n_total  rows  iters  status")
    print("-" * 92)
    for backend in backends:
        row = measure_backend(
            backend=backend,
            problem=problem,
            urdf_path=urdf_path,
            order=int(args.order),
            repeat=int(args.repeat),
            warmup=int(args.warmup),
            solve_repeat=int(args.solve_repeat),
        )
        print(
            f"{backend:15s}"
            f"{row['load_ms']:9.3f}"
            f"{row['compile_ms']:12.3f}"
            f"{row['linearize_ms']:14.3f}"
            f"{row['solve_ms']:10.3f}"
            f"{row['n_total']:9d}"
            f"{row['rows']:6d}"
            f"{row['iters']:7d}  "
            f"{row['status']}"
        )


if __name__ == "__main__":
    main()
