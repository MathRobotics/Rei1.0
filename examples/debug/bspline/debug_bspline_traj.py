from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This debug script requires matplotlib.\n"
        "Install dependencies and re-run:\n"
        "  python -m pip install -e ."
    ) from e

from eiopt import load_problem_toml
from eiopt.core.bspline import (
    bspline_basis_derivative_matrix,
    bspline_basis_matrix,
    default_clamped_uniform_knots,
)
from eiopt.core.trajectory import TrajectoryMap
from eiopt.dsl.trajectory import (
    build_trajectory_map,
    build_trajectory_map_with_derivative,
    default_dt_from_time,
    default_steps_from_time,
    infer_bspline_q_dim_from_var,
    pick_trajectory_value,
)


def _resolve_positive_int(value: object, *, name: str) -> int:
    try:
        out = int(value)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name} must be an integer, got {value!r}.") from e
    if out <= 0:
        raise ValueError(f"{name} must be > 0, got {out}.")
    return out


def _resolve_nonnegative_int(value: object, *, name: str) -> int:
    try:
        out = int(value)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name} must be an integer, got {value!r}.") from e
    if out < 0:
        raise ValueError(f"{name} must be >= 0, got {out}.")
    return out


def _resolve_positive_float(value: object, *, name: str) -> float:
    try:
        out = float(value)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name} must be a float, got {value!r}.") from e
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {out}.")
    return out


def _find_var_dsl(dsl: Mapping[str, object], *, name: str) -> Mapping[str, object] | None:
    vars_dsl = dsl.get("variables", None)
    if not isinstance(vars_dsl, list):
        return None
    for v in vars_dsl:
        if isinstance(v, Mapping) and str(v.get("name", "")).strip() == name:
            return v
    return None


def _demo_control_points(*, num_ctrl_points: int, q_dim: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, int(num_ctrl_points), dtype=float)
    cp = np.zeros((int(num_ctrl_points), int(q_dim)), dtype=float)
    if q_dim >= 1:
        cp[:, 0] = u
    if q_dim >= 2:
        cp[:, 1] = 0.5 * np.sin(2.0 * np.pi * u)
    if q_dim >= 3:
        cp[:, 2] = 0.25 * np.cos(2.0 * np.pi * u)
    for j in range(3, q_dim):
        cp[:, j] = 0.1 * float(j - 2) * u
    return cp.reshape(-1)


def _load_or_generate_p(
    root_dsl: Mapping[str, object],
    *,
    var_name: str,
    p_dim: int,
    num_ctrl_points: int,
    q_dim: int,
) -> tuple[np.ndarray, str]:
    var_dsl = _find_var_dsl(root_dsl, name=var_name)
    if var_dsl is not None and "init" in var_dsl:
        init = np.asarray(var_dsl.get("init", []), dtype=float).reshape(-1)
        if init.size == int(p_dim):
            if np.max(np.abs(init)) > 1e-12:
                return init, f"DSL variable init ({var_name})"
            return _demo_control_points(num_ctrl_points=num_ctrl_points, q_dim=q_dim), (
                f"auto-generated demo control points "
                f"(DSL init for {var_name!r} is all zeros)"
            )

    return _demo_control_points(num_ctrl_points=num_ctrl_points, q_dim=q_dim), (
        "auto-generated demo control points "
        "(variable init not found or dimension mismatch)"
    )


def _build_bspline_debug_data(
    *,
    root_dsl: Mapping[str, object],
    traj_dsl: Mapping[str, object],
    steps: int | None,
    q_dim: int | None,
) -> dict[str, np.ndarray | int | str]:
    typ = str(traj_dsl.get("type", "")).strip().lower()
    if typ != "bspline":
        raise ValueError(f"trajectory.type must be 'bspline', got {typ!r}.")

    degree_raw = pick_trajectory_value(traj_dsl, section="bspline", key="degree")
    nctrl_raw = pick_trajectory_value(traj_dsl, section="bspline", key="num_ctrl_points")
    if degree_raw is None or nctrl_raw is None:
        raise ValueError("trajectory.degree and trajectory.num_ctrl_points are required.")

    degree = _resolve_nonnegative_int(degree_raw, name="degree")
    num_ctrl_points = _resolve_positive_int(nctrl_raw, name="num_ctrl_points")
    var_name = str(traj_dsl.get("var", "p")).strip() or "p"

    default_steps = int(steps) if steps is not None else default_steps_from_time(root_dsl)
    if default_steps is None:
        raise ValueError("Could not infer steps. Set trajectory.steps, [time].N, or pass --steps.")

    default_q_dim = None
    if q_dim is not None:
        default_q_dim = int(q_dim)
    else:
        var_dsl = _find_var_dsl(root_dsl, name=var_name)
        if var_dsl is not None and "dim" in var_dsl:
            var_dim = _resolve_positive_int(var_dsl["dim"], name=f"variables[{var_name!r}].dim")
            default_q_dim = infer_bspline_q_dim_from_var(traj_dsl, var_dim=var_dim)

    traj = build_trajectory_map(
        traj_dsl,
        default_steps=default_steps,
        default_q_dim=default_q_dim,
    )
    steps_res = int(traj.steps)
    q_dim_res = int(traj.q_dim)

    knot_vector_raw = pick_trajectory_value(traj_dsl, section="bspline", key="knot_vector")
    u_samples_raw = pick_trajectory_value(traj_dsl, section="bspline", key="u_samples")
    knots = (
        default_clamped_uniform_knots(num_ctrl_points=num_ctrl_points, degree=degree)
        if knot_vector_raw is None
        else np.asarray(knot_vector_raw, dtype=float).reshape(-1)
    )
    if u_samples_raw is None:
        u_min = float(knots[degree])
        u_max = float(knots[num_ctrl_points])
        u_samples = np.linspace(u_min, u_max, steps_res, dtype=float)
    else:
        u_samples = np.asarray(u_samples_raw, dtype=float).reshape(-1)

    basis = bspline_basis_matrix(
        u_vec=u_samples,
        degree=degree,
        knots=knots,
        num_ctrl_points=num_ctrl_points,
    )

    p, p_source = _load_or_generate_p(
        root_dsl,
        var_name=var_name,
        p_dim=traj.p_dim,
        num_ctrl_points=num_ctrl_points,
        q_dim=q_dim_res,
    )
    q_traj = np.vstack([traj.q_at(p, k) for k in range(traj.steps)])
    ctrl = p.reshape(num_ctrl_points, q_dim_res)
    row_sums = np.sum(basis, axis=1)

    return {
        "traj": traj,
        "degree": degree,
        "num_ctrl_points": num_ctrl_points,
        "steps": steps_res,
        "q_dim": q_dim_res,
        "var_name": var_name,
        "knots": knots,
        "u_samples": u_samples,
        "basis": basis,
        "p": p,
        "p_source": p_source,
        "ctrl": ctrl,
        "q_traj": q_traj,
        "row_sums": row_sums,
    }


def _check_jacobian(traj: TrajectoryMap, p: np.ndarray, *, eps: float) -> float:
    p_vec = np.asarray(p, dtype=float).reshape(-1)
    max_abs = 0.0
    for k in range(traj.steps):
        j_analytic = traj.dqdp_at(k)
        q0 = traj.q_at(p_vec, k)
        j_fd = np.zeros_like(j_analytic)
        for col in range(traj.p_dim):
            dp = np.zeros((traj.p_dim,), dtype=float)
            dp[col] = float(eps)
            q1 = traj.q_at(p_vec + dp, k)
            j_fd[:, col] = (q1 - q0) / float(eps)
        max_abs = max(max_abs, float(np.max(np.abs(j_analytic - j_fd))))
    return max_abs


def _resolve_bspline_scale(
    *,
    derivative_order: int,
    derivative_wrt: str,
    degree: int,
    num_ctrl_points: int,
    knots: np.ndarray,
    u_samples: np.ndarray,
    steps: int,
    dt: float | None,
) -> float:
    if derivative_order <= 0:
        return 1.0
    wrt = str(derivative_wrt).strip().lower()
    if wrt in ("u", "param", "parameter"):
        return 1.0
    if wrt != "time":
        raise ValueError(f"Unsupported derivative_wrt: {derivative_wrt!r}")

    if dt is None:
        raise ValueError("dt is required when derivative_wrt='time'.")
    dt_val = _resolve_positive_float(dt, name="dt")
    if int(steps) <= 1:
        raise ValueError("steps must be >= 2 when derivative_wrt='time'.")

    u_span = float(knots[int(num_ctrl_points)] - knots[int(degree)])
    if u_samples.size >= 2:
        du = np.diff(u_samples)
        if np.allclose(du, du[0], atol=1e-10, rtol=1e-9):
            u_span = float(u_samples[-1] - u_samples[0])

    horizon = float((int(steps) - 1) * dt_val)
    if horizon <= 0.0:
        raise ValueError("invalid time horizon.")
    scale = u_span / horizon
    return float(scale**int(derivative_order))


def _build_bspline_derivative_debug_data(
    *,
    root_dsl: Mapping[str, object],
    traj_dsl: Mapping[str, object],
    p: np.ndarray,
    steps: int,
    q_dim: int,
    degree: int,
    num_ctrl_points: int,
    knots: np.ndarray,
    u_samples: np.ndarray,
    derivative_order: int,
    derivative_wrt: str,
    dt_override: float | None,
) -> dict[str, np.ndarray | int | str | float]:
    dt_default = default_dt_from_time(root_dsl)
    dt = dt_default if dt_override is None else float(dt_override)

    traj_d = build_trajectory_map_with_derivative(
        traj_dsl,
        derivative_order=derivative_order,
        derivative_wrt=derivative_wrt,
        default_steps=steps,
        default_q_dim=q_dim,
        default_dt=dt,
    )
    qd_traj = np.vstack([traj_d.q_at(p, k) for k in range(traj_d.steps)])
    basis_d = bspline_basis_derivative_matrix(
        u_vec=u_samples,
        degree=degree,
        knots=knots,
        num_ctrl_points=num_ctrl_points,
        derivative_order=derivative_order,
    )
    scale = _resolve_bspline_scale(
        derivative_order=derivative_order,
        derivative_wrt=derivative_wrt,
        degree=degree,
        num_ctrl_points=num_ctrl_points,
        knots=knots,
        u_samples=u_samples,
        steps=steps,
        dt=dt,
    )
    basis_d_scaled = scale * basis_d
    return {
        "traj_d": traj_d,
        "q_d": qd_traj,
        "basis_d": basis_d_scaled,
        "row_sums_d": np.sum(basis_d_scaled, axis=1),
        "dt": float(dt) if dt is not None else np.nan,
        "derivative_order": int(derivative_order),
        "derivative_wrt": str(derivative_wrt),
    }


def _plot_debug_figure(data: dict[str, np.ndarray | int | str], *, output: Path, show: bool) -> None:
    ctrl = np.asarray(data["ctrl"], dtype=float)
    q_traj = np.asarray(data["q_traj"], dtype=float)
    basis = np.asarray(data["basis"], dtype=float)
    u_samples = np.asarray(data["u_samples"], dtype=float)
    row_sums = np.asarray(data["row_sums"], dtype=float)
    q_dim = int(data["q_dim"])
    degree = int(data["degree"])
    n_ctrl = int(data["num_ctrl_points"])
    steps = int(data["steps"])

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))

    ax0 = axes[0, 0]
    if q_dim >= 2:
        ax0.plot(ctrl[:, 0], ctrl[:, 1], "o--", label="control polygon")
        ax0.plot(q_traj[:, 0], q_traj[:, 1], "o-", label="sampled trajectory")
        for k in range(steps):
            ax0.text(float(q_traj[k, 0]), float(q_traj[k, 1]), f"k={k}", fontsize=8)
        ax0.set_xlabel("q[0]")
        ax0.set_ylabel("q[1]")
        ax0.axis("equal")
    else:
        k = np.arange(steps, dtype=int)
        ax0.plot(k, q_traj[:, 0], "o-", label="q(k)")
        ax0.plot(np.linspace(0, steps - 1, n_ctrl), ctrl[:, 0], "x--", label="control points")
        ax0.set_xlabel("sample index k")
        ax0.set_ylabel("q[0]")
    ax0.set_title("Trajectory")
    ax0.grid(True, alpha=0.35)
    ax0.legend()

    ax1 = axes[0, 1]
    ks = np.arange(steps, dtype=int)
    for j in range(q_dim):
        ax1.plot(ks, q_traj[:, j], "o-", label=f"q[{j}]")
    ax1.set_title("Each Dimension vs k")
    ax1.set_xlabel("k")
    ax1.set_ylabel("q")
    ax1.grid(True, alpha=0.35)
    ax1.legend()

    ax2 = axes[1, 0]
    for i in range(n_ctrl):
        ax2.plot(u_samples, basis[:, i], label=f"N{i}")
    ax2.set_title("B-spline Basis")
    ax2.set_xlabel("u")
    ax2.set_ylabel("N_i(u)")
    ax2.grid(True, alpha=0.35)
    ax2.legend(ncol=2, fontsize=8)

    ax3 = axes[1, 1]
    im = ax3.imshow(basis, aspect="auto", origin="lower")
    ax3.set_title("Basis Matrix (k x control idx)")
    ax3.set_xlabel("control point index")
    ax3.set_ylabel("sample index k")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("weight")

    fig.suptitle(
        f"B-spline debug: degree={degree}, ctrl={n_ctrl}, steps={steps}, q_dim={q_dim}",
        fontsize=12,
    )
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    if show:
        plt.show()
    plt.close(fig)

    print(f"[saved] {output}")
    print(
        "[basis-row-sum] min={:.12f}, max={:.12f}, mean={:.12f}".format(
            float(np.min(row_sums)),
            float(np.max(row_sums)),
            float(np.mean(row_sums)),
        )
    )


def _plot_derivative_debug_figure(
    data: dict[str, np.ndarray | int | str | float], *, output: Path, show: bool
) -> None:
    q_d = np.asarray(data["q_d"], dtype=float)
    basis_d = np.asarray(data["basis_d"], dtype=float)
    row_sums_d = np.asarray(data["row_sums_d"], dtype=float)
    order = int(data["derivative_order"])
    wrt = str(data["derivative_wrt"])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))

    ax0 = axes[0]
    ks = np.arange(q_d.shape[0], dtype=int)
    for j in range(q_d.shape[1]):
        ax0.plot(ks, q_d[:, j], "o-", label=f"d^{order}q[{j}]/d{wrt}^{order}")
    ax0.set_title("Derivative Trajectory")
    ax0.set_xlabel("k")
    ax0.set_ylabel("value")
    ax0.grid(True, alpha=0.35)
    ax0.legend()

    ax1 = axes[1]
    for i in range(basis_d.shape[1]):
        ax1.plot(np.arange(basis_d.shape[0]), basis_d[:, i], label=f"dN{i}")
    ax1.set_title("Derivative Basis (sample index)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("weight")
    ax1.grid(True, alpha=0.35)
    ax1.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    if show:
        plt.show()
    plt.close(fig)

    print(f"[saved] {output}")
    print(
        "[basis-derivative-row-sum] min={:.12f}, max={:.12f}, mean={:.12f}".format(
            float(np.min(row_sums_d)),
            float(np.max(row_sums_d)),
            float(np.mean(row_sums_d)),
        )
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug plot utility for TrajectoryMap.from_bspline.")
    parser.add_argument(
        "--dsl",
        type=Path,
        default=Path("examples/dsl/kots_traj_pos.toml"),
        help="Path to TOML with [trajectory] type='bspline'.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override trajectory steps.")
    parser.add_argument("--q-dim", type=int, default=None, help="Override trajectory q_dim.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/debug/bspline/out/bspline_debug.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot window in addition to saving.")
    parser.add_argument(
        "--check-jacobian",
        action="store_true",
        help="Run finite-difference Jacobian check and print max absolute error.",
    )
    parser.add_argument(
        "--fd-eps",
        type=float,
        default=1e-7,
        help="Finite-difference epsilon for --check-jacobian.",
    )
    parser.add_argument(
        "--check-derivative",
        action="store_true",
        help="Build derivative trajectory map and run finite-difference Jacobian check.",
    )
    parser.add_argument(
        "--derivative-order",
        type=int,
        default=1,
        help="Derivative order for --check-derivative (>=1).",
    )
    parser.add_argument(
        "--derivative-wrt",
        type=str,
        choices=["u", "time"],
        default="u",
        help="Derivative axis for --check-derivative.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override dt when --derivative-wrt=time (default: read [time].dt).",
    )
    parser.add_argument(
        "--derivative-output",
        type=Path,
        default=None,
        help="Output PNG path for derivative debug plot.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    root_dsl = load_problem_toml(args.dsl)
    traj_dsl = root_dsl.get("trajectory", None)
    if not isinstance(traj_dsl, Mapping):
        raise SystemExit("DSL must contain [trajectory] section.")

    data = _build_bspline_debug_data(
        root_dsl=root_dsl,
        traj_dsl=traj_dsl,
        steps=args.steps,
        q_dim=args.q_dim,
    )
    print(
        "[config] steps={steps}, q_dim={q_dim}, degree={degree}, num_ctrl_points={nctrl}, p_source={p_source}".format(
            steps=int(data["steps"]),
            q_dim=int(data["q_dim"]),
            degree=int(data["degree"]),
            nctrl=int(data["num_ctrl_points"]),
            p_source=str(data["p_source"]),
        )
    )
    print(f"[knots] {np.asarray(data['knots'], dtype=float)}")
    print(f"[u_samples] {np.asarray(data['u_samples'], dtype=float)}")

    _plot_debug_figure(data, output=args.output, show=bool(args.show))

    if args.check_jacobian:
        traj = data["traj"]
        if not isinstance(traj, TrajectoryMap):  # pragma: no cover
            raise RuntimeError("Internal error: invalid trajectory object.")
        p = np.asarray(data["p"], dtype=float).reshape(-1)
        max_abs = _check_jacobian(traj, p, eps=float(args.fd_eps))
        print(f"[jacobian-check] max_abs_error={max_abs:.6e}, fd_eps={float(args.fd_eps):.1e}")

    if args.check_derivative:
        d_order = _resolve_positive_int(args.derivative_order, name="--derivative-order")
        p = np.asarray(data["p"], dtype=float).reshape(-1)
        deriv_data = _build_bspline_derivative_debug_data(
            root_dsl=root_dsl,
            traj_dsl=traj_dsl,
            p=p,
            steps=int(data["steps"]),
            q_dim=int(data["q_dim"]),
            degree=int(data["degree"]),
            num_ctrl_points=int(data["num_ctrl_points"]),
            knots=np.asarray(data["knots"], dtype=float).reshape(-1),
            u_samples=np.asarray(data["u_samples"], dtype=float).reshape(-1),
            derivative_order=d_order,
            derivative_wrt=str(args.derivative_wrt),
            dt_override=args.dt,
        )
        print(
            "[derivative-config] order={order}, wrt={wrt}, dt={dt}".format(
                order=int(deriv_data["derivative_order"]),
                wrt=str(deriv_data["derivative_wrt"]),
                dt=float(deriv_data["dt"]),
            )
        )

        deriv_output = args.derivative_output
        if deriv_output is None:
            deriv_output = args.output.with_name(f"{args.output.stem}_d{d_order}_{args.derivative_wrt}.png")
        _plot_derivative_debug_figure(deriv_data, output=deriv_output, show=bool(args.show))

        traj_d = deriv_data["traj_d"]
        if not isinstance(traj_d, TrajectoryMap):  # pragma: no cover
            raise RuntimeError("Internal error: invalid derivative trajectory object.")
        max_abs_d = _check_jacobian(traj_d, p, eps=float(args.fd_eps))
        print(f"[derivative-jacobian-check] max_abs_error={max_abs_d:.6e}, fd_eps={float(args.fd_eps):.1e}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
