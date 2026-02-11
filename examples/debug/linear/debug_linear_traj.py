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
from eiopt.core.trajectory import TrajectoryMap
from eiopt.dsl.trajectory import build_trajectory_map, default_steps_from_time


def _resolve_positive_int(value: object, *, name: str) -> int:
    try:
        out = int(value)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name} must be an integer, got {value!r}.") from e
    if out <= 0:
        raise ValueError(f"{name} must be > 0, got {out}.")
    return out


def _find_var_dsl(dsl: Mapping[str, object], *, name: str) -> Mapping[str, object] | None:
    vars_dsl = dsl.get("variables", None)
    if not isinstance(vars_dsl, list):
        return None
    for var in vars_dsl:
        if isinstance(var, Mapping) and str(var.get("name", "")).strip() == name:
            return var
    return None


def _parse_cli_p(raw: str | None) -> np.ndarray | None:
    if raw is None:
        return None
    text = raw.strip()
    if text == "":
        raise ValueError("--p must not be empty.")
    parts = [p.strip() for p in text.split(",")]
    if any(p == "" for p in parts):
        raise ValueError("--p must be comma-separated numeric values (e.g. 0.0,1.0,-0.3).")
    try:
        return np.asarray([float(p) for p in parts], dtype=float).reshape(-1)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Failed to parse --p values: {raw!r}.") from e


def _make_demo_p(p_dim: int) -> np.ndarray:
    p = np.linspace(-0.25, 0.25, int(p_dim), dtype=float)
    if p_dim >= 2:
        p[0] = 0.0
        p[1] = 0.0
    if p_dim >= 4:
        p[2] = 1.0
        p[3] = 0.8
    return p


def _load_or_generate_p(
    root_dsl: Mapping[str, object],
    *,
    var_name: str,
    p_dim: int,
    p_override: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    if p_override is not None:
        p = np.asarray(p_override, dtype=float).reshape(-1)
        if p.size != int(p_dim):
            raise ValueError(f"--p size mismatch. Expected {p_dim}, got {p.size}.")
        return p, "CLI --p"

    var_dsl = _find_var_dsl(root_dsl, name=var_name)
    if var_dsl is not None and "init" in var_dsl:
        init = np.asarray(var_dsl.get("init", []), dtype=float).reshape(-1)
        if init.size == int(p_dim):
            if float(np.max(np.abs(init))) > 1e-12:
                return init, f"DSL variable init ({var_name})"
            return _make_demo_p(p_dim), f"auto-generated demo p (DSL init for {var_name!r} is all zeros)"
    return _make_demo_p(p_dim), "auto-generated demo p (variable init not found or dimension mismatch)"


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


def _plot_debug_figure(
    *,
    traj: TrajectoryMap,
    p: np.ndarray,
    output: Path,
    show: bool,
) -> None:
    a = np.asarray(traj.A, dtype=float)
    b = np.asarray(traj.b, dtype=float).reshape(-1)
    q = (a @ p + b).reshape(traj.steps, traj.q_dim)
    b_blocks = b.reshape(traj.steps, traj.q_dim)
    ks = np.arange(traj.steps, dtype=int)
    a_norms = np.asarray([np.linalg.norm(traj.dqdp_at(k), ord="fro") for k in range(traj.steps)], dtype=float)
    b_norms = np.linalg.norm(b_blocks, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))

    ax0 = axes[0, 0]
    if traj.q_dim >= 2:
        ax0.plot(q[:, 0], q[:, 1], "o-", label="trajectory")
        for k in range(traj.steps):
            ax0.text(float(q[k, 0]), float(q[k, 1]), f"k={k}", fontsize=8)
        ax0.set_xlabel("q[0]")
        ax0.set_ylabel("q[1]")
        ax0.axis("equal")
    else:
        ax0.plot(ks, q[:, 0], "o-", label="q[0]")
        ax0.set_xlabel("k")
        ax0.set_ylabel("q[0]")
    ax0.set_title("Trajectory")
    ax0.grid(True, alpha=0.35)
    ax0.legend()

    ax1 = axes[0, 1]
    for j in range(traj.q_dim):
        ax1.plot(ks, q[:, j], "o-", label=f"q[{j}]")
        if np.max(np.abs(b_blocks[:, j])) > 1e-14:
            ax1.plot(ks, b_blocks[:, j], "--", label=f"b[{j}]")
    ax1.set_title("Each Dimension vs k")
    ax1.set_xlabel("k")
    ax1.set_ylabel("q")
    ax1.grid(True, alpha=0.35)
    ax1.legend()

    ax2 = axes[1, 0]
    im = ax2.imshow(a, aspect="auto", origin="lower")
    ax2.set_title("Linear Map A (rows=stacked q, cols=p)")
    ax2.set_xlabel("p index")
    ax2.set_ylabel("stacked q row")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("A value")

    ax3 = axes[1, 1]
    ax3.plot(ks, a_norms, "o-", label="||A_k||_F")
    ax3.plot(ks, b_norms, "s--", label="||b_k||_2")
    ax3.set_title("Per-step Block Norms")
    ax3.set_xlabel("k")
    ax3.set_ylabel("norm")
    ax3.grid(True, alpha=0.35)
    ax3.legend()

    fig.suptitle(
        f"Linear trajectory debug: steps={traj.steps}, q_dim={traj.q_dim}, p_dim={traj.p_dim}",
        fontsize=12,
    )
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    if show:
        plt.show()
    plt.close(fig)

    print(f"[saved] {output}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug plot utility for linear trajectory maps.")
    parser.add_argument(
        "--dsl",
        type=Path,
        default=Path("examples/debug/linear/linear_traj_demo.toml"),
        help="Path to TOML with [trajectory] type='linear'.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override default trajectory steps.")
    parser.add_argument("--q-dim", type=int, default=None, help="Override default trajectory q_dim.")
    parser.add_argument(
        "--p",
        type=str,
        default=None,
        help="Optional p values, comma-separated (example: 0.0,0.0,1.0,0.8).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/debug/linear/out/linear_debug.png"),
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
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    root_dsl = load_problem_toml(args.dsl)
    traj_dsl = root_dsl.get("trajectory", None)
    if not isinstance(traj_dsl, Mapping):
        raise SystemExit("DSL must contain [trajectory] section.")
    if str(traj_dsl.get("type", "")).strip().lower() != "linear":
        raise SystemExit(f"trajectory.type must be 'linear', got {traj_dsl.get('type', None)!r}.")

    default_steps = _resolve_positive_int(args.steps, name="--steps") if args.steps is not None else default_steps_from_time(root_dsl)
    default_q_dim = _resolve_positive_int(args.q_dim, name="--q-dim") if args.q_dim is not None else None

    traj = build_trajectory_map(traj_dsl, default_steps=default_steps, default_q_dim=default_q_dim)
    var_name = str(traj_dsl.get("var", "p")).strip() or "p"
    p_override = _parse_cli_p(args.p)
    p, p_source = _load_or_generate_p(root_dsl, var_name=var_name, p_dim=traj.p_dim, p_override=p_override)

    q_stack = (traj.A @ p + traj.b).reshape(traj.steps, traj.q_dim)
    q_eval = np.vstack([traj.q_at(p, k) for k in range(traj.steps)])
    eval_diff = float(np.max(np.abs(q_stack - q_eval)))

    a = np.asarray(traj.A, dtype=float)
    rank = int(np.linalg.matrix_rank(a))
    svals = np.linalg.svd(a, compute_uv=False)
    cond = float(svals[0] / svals[-1]) if svals[-1] > 0.0 else float("inf")

    print(
        "[config] steps={steps}, q_dim={q_dim}, p_dim={p_dim}, A_shape={a_shape}, b_shape={b_shape}, p_source={p_source}".format(
            steps=traj.steps,
            q_dim=traj.q_dim,
            p_dim=traj.p_dim,
            a_shape=tuple(a.shape),
            b_shape=tuple(np.asarray(traj.b).shape),
            p_source=p_source,
        )
    )
    print(f"[matrix] rank(A)={rank}, cond_est(A)={cond:.6e}")
    print(f"[consistency] max_abs((A@p+b)-q_at)={eval_diff:.6e}")
    print(f"[p] {p}")
    print(f"[q(0)] {q_eval[0]}")
    print(f"[q(end)] {q_eval[-1]}")

    _plot_debug_figure(traj=traj, p=p, output=args.output, show=bool(args.show))

    if args.check_jacobian:
        max_abs = _check_jacobian(traj, p, eps=float(args.fd_eps))
        print(f"[jacobian-check] max_abs_error={max_abs:.6e}, fd_eps={float(args.fd_eps):.1e}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
