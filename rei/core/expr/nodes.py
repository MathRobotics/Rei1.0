from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..state_cache import StateKey
from ..trajectory import TrajectoryMap
from .types import Expr, RuntimeContext, Variable


@dataclass
class GetStateExpr:
    name: str
    vars: Sequence[Variable]
    key_value: StateKey
    key_jacs: Sequence[StateKey]

    def deps(self):
        return [self.key_value, *self.key_jacs]

    def value_deps(self):
        return [self.key_value]

    def eval_value(self, ctx: RuntimeContext):
        sc = ctx.state  # StateCache-like
        return np.asarray(sc.get(self.key_value), dtype=float).reshape(-1)

    def eval(self, ctx: RuntimeContext):
        sc = ctx.state  # StateCache-like
        y = self.eval_value(ctx)

        if len(self.vars) != len(self.key_jacs):
            raise ValueError(
                f"{self.name}: internal mismatch len(vars)={len(self.vars)} vs len(key_jacs)={len(self.key_jacs)}."
            )

        blocks = []
        for v, key_jac in zip(self.vars, self.key_jacs):
            J = np.asarray(sc.get(key_jac), dtype=float)
            if J.shape != (y.size, v.dim()):
                raise ValueError(f"{self.name}: J shape mismatch for var '{v.name}': {J.shape} vs {(y.size, v.dim())}")
            blocks.append(J)
        return y, blocks

    def vjp(self, ctx: RuntimeContext, rhs):
        sc = ctx.state  # StateCache-like
        y = np.asarray(sc.get(self.key_value), dtype=float).reshape(-1)
        r = np.asarray(rhs, dtype=float)
        if r.ndim not in (1, 2) or int(r.shape[0]) != int(y.size):
            raise ValueError(f"{self.name}: rhs shape mismatch for vjp: rhs={r.shape}, value size={y.size}.")

        if len(self.vars) != len(self.key_jacs):
            raise ValueError(
                f"{self.name}: internal mismatch len(vars)={len(self.vars)} vs len(key_jacs)={len(self.key_jacs)}."
            )

        fast_vjp = getattr(sc, "jacobian_transpose_mul", None)
        out = []
        for v, key_jac in zip(self.vars, self.key_jacs):
            g = None
            if callable(fast_vjp):
                try:
                    g = np.asarray(fast_vjp(self.key_value, key_jac, r), dtype=float)
                except (AttributeError, KeyError, ValueError, TypeError, RuntimeError):
                    g = None
            if g is None:
                J = np.asarray(sc.get(key_jac), dtype=float)
                if J.shape != (y.size, v.dim()):
                    raise ValueError(
                        f"{self.name}: J shape mismatch for var '{v.name}': {J.shape} vs {(y.size, v.dim())}"
                    )
                g = np.asarray(J.T @ r, dtype=float)

            expected = (v.dim(),) if r.ndim == 1 else (v.dim(), int(r.shape[1]))
            if g.shape != expected:
                g = np.asarray(g, dtype=float).reshape(expected)
            out.append(g)
        return out


@dataclass
class GetVarExpr:
    """Read decision variables directly from the VariablePack (no StateCache deps).

    If `k` is provided and the variable dimension is divisible by (time.N+1),
    the variable is treated as a stacked trajectory and sliced by k.
    """

    name: str
    vars: Sequence[Variable]
    k: int | None = None

    def deps(self):
        return []

    def value_deps(self):
        return []

    def eval_value(self, ctx: RuntimeContext):
        x = np.asarray(self.vars[0].x, dtype=float).reshape(-1)
        n_total = int(x.size)

        if self.k is None:
            return x.copy()

        k = int(self.k)

        steps = 1
        time = getattr(ctx, "time", None)
        if time is not None and hasattr(time, "N"):
            try:
                steps = int(time.N) + 1
            except Exception:
                steps = 1

        chunked = bool(steps > 1 and n_total % steps == 0)
        if not chunked:
            if k != 0:
                raise ValueError(
                    f"{self.name}: requested k={k}, but variable '{self.vars[0].name}' is not time-chunked "
                    f"(dim={n_total}, steps={steps})."
                )
            return x.copy()

        if k < 0 or k >= steps:
            raise ValueError(f"{self.name}: requested k={k}, but time steps are 0..{steps - 1}.")

        n = int(n_total // steps)
        start = int(k * n)
        return x[start : start + n].copy()

    def eval(self, ctx: RuntimeContext):
        x = np.asarray(self.vars[0].x, dtype=float).reshape(-1)
        n_total = int(x.size)

        if self.k is None:
            return x.copy(), [np.eye(n_total, dtype=float)]

        k = int(self.k)

        steps = 1
        time = getattr(ctx, "time", None)
        if time is not None and hasattr(time, "N"):
            try:
                steps = int(time.N) + 1
            except Exception:
                steps = 1

        chunked = bool(steps > 1 and n_total % steps == 0)
        if not chunked:
            if k != 0:
                raise ValueError(
                    f"{self.name}: requested k={k}, but variable '{self.vars[0].name}' is not time-chunked "
                    f"(dim={n_total}, steps={steps})."
                )
            return x.copy(), [np.eye(n_total, dtype=float)]

        if k < 0 or k >= steps:
            raise ValueError(f"{self.name}: requested k={k}, but time steps are 0..{steps - 1}.")

        n = int(n_total // steps)
        start = int(k * n)
        y = x[start : start + n].copy()

        J = np.zeros((n, n_total), dtype=float)
        J[:, start : start + n] = np.eye(n, dtype=float)
        return y, [J]


@dataclass
class TrajectoryVarExpr:
    """Map trajectory parameter variable to stacked trajectory `A @ p + b`."""

    name: str
    vars: Sequence[Variable]
    trajectory: TrajectoryMap
    k: int | None = None

    def deps(self):
        return []

    def value_deps(self):
        return []

    def eval_value(self, ctx: RuntimeContext):
        del ctx
        p = np.asarray(self.vars[0].x, dtype=float).reshape(-1)
        if p.size != self.trajectory.p_dim:
            raise ValueError(
                f"{self.name}: parameter size mismatch. "
                f"Expected {self.trajectory.p_dim}, got {p.size}."
            )
        y_all = (self.trajectory.A @ p + self.trajectory.b).reshape(-1)

        if self.k is None:
            return y_all

        k = int(self.k)
        steps = int(self.trajectory.steps)
        if k < 0 or k >= steps:
            raise ValueError(f"{self.name}: requested k={k}, but time steps are 0..{steps - 1}.")

        seg = int(self.trajectory.q_dim)
        start = int(k * seg)
        stop = int(start + seg)
        return y_all[start:stop].copy()

    def eval(self, ctx: RuntimeContext):
        del ctx
        p = np.asarray(self.vars[0].x, dtype=float).reshape(-1)
        if p.size != self.trajectory.p_dim:
            raise ValueError(
                f"{self.name}: parameter size mismatch. "
                f"Expected {self.trajectory.p_dim}, got {p.size}."
            )
        y_all = (self.trajectory.A @ p + self.trajectory.b).reshape(-1)

        if self.k is None:
            return y_all, [self.trajectory.A.copy()]

        k = int(self.k)
        steps = int(self.trajectory.steps)
        if k < 0 or k >= steps:
            raise ValueError(f"{self.name}: requested k={k}, but time steps are 0..{steps - 1}.")

        seg = int(self.trajectory.q_dim)
        start = int(k * seg)
        stop = int(start + seg)
        return y_all[start:stop].copy(), [self.trajectory.A[start:stop, :].copy()]


@dataclass
class TrajectoryVarDerivativesExpr:
    """Map trajectory parameter variable to stacked derivatives ``[q, dq, ..., d^Nq]``."""

    name: str
    vars: Sequence[Variable]
    trajectories: Sequence[TrajectoryMap]
    k: int | None = None

    def deps(self):
        return []

    def value_deps(self):
        return []

    def eval_value(self, ctx: RuntimeContext):
        del ctx
        if len(self.trajectories) == 0:
            raise ValueError(f"{self.name}: trajectories must be non-empty.")

        p = np.asarray(self.vars[0].x, dtype=float).reshape(-1)
        p_dim = int(self.trajectories[0].p_dim)
        if p.size != p_dim:
            raise ValueError(
                f"{self.name}: parameter size mismatch. "
                f"Expected {p_dim}, got {p.size}."
            )

        steps = int(self.trajectories[0].steps)
        q_dim = int(self.trajectories[0].q_dim)
        for i, traj in enumerate(self.trajectories):
            if traj.p_dim != p_dim:
                raise ValueError(
                    f"{self.name}: trajectory[{i}] p_dim mismatch. "
                    f"Expected {p_dim}, got {traj.p_dim}."
                )
            if traj.steps != steps or traj.q_dim != q_dim:
                raise ValueError(
                    f"{self.name}: trajectory[{i}] shape mismatch. "
                    f"Expected steps={steps}, q_dim={q_dim}, got steps={traj.steps}, q_dim={traj.q_dim}."
                )

        if self.k is not None:
            k = int(self.k)
            if k < 0 or k >= steps:
                raise ValueError(f"{self.name}: requested k={k}, but time steps are 0..{steps - 1}.")

            start = int(k * q_dim)
            stop = int(start + q_dim)
            y_parts = []
            for traj in self.trajectories:
                y_all = (traj.A @ p + traj.b).reshape(-1)
                y_parts.append(y_all[start:stop].copy())
            return np.concatenate(y_parts, axis=0)

        return np.concatenate(
            [(traj.A @ p + traj.b).reshape(-1) for traj in self.trajectories],
            axis=0,
        )

    def eval(self, ctx: RuntimeContext):
        if len(self.trajectories) == 0:
            raise ValueError(f"{self.name}: trajectories must be non-empty.")

        p = np.asarray(self.vars[0].x, dtype=float).reshape(-1)
        p_dim = int(self.trajectories[0].p_dim)
        if p.size != p_dim:
            raise ValueError(
                f"{self.name}: parameter size mismatch. "
                f"Expected {p_dim}, got {p.size}."
            )

        steps = int(self.trajectories[0].steps)
        q_dim = int(self.trajectories[0].q_dim)
        for i, traj in enumerate(self.trajectories):
            if traj.p_dim != p_dim:
                raise ValueError(
                    f"{self.name}: trajectory[{i}] p_dim mismatch. "
                    f"Expected {p_dim}, got {traj.p_dim}."
                )
            if traj.steps != steps or traj.q_dim != q_dim:
                raise ValueError(
                    f"{self.name}: trajectory[{i}] shape mismatch. "
                    f"Expected steps={steps}, q_dim={q_dim}, got steps={traj.steps}, q_dim={traj.q_dim}."
                )

        if self.k is not None:
            k = int(self.k)
            if k < 0 or k >= steps:
                raise ValueError(f"{self.name}: requested k={k}, but time steps are 0..{steps - 1}.")

            start = int(k * q_dim)
            stop = int(start + q_dim)
            y_parts = []
            j_parts = []
            for traj in self.trajectories:
                y_all = (traj.A @ p + traj.b).reshape(-1)
                y_parts.append(y_all[start:stop].copy())
                j_parts.append(traj.A[start:stop, :].copy())
            return np.concatenate(y_parts, axis=0), [np.vstack(j_parts)]

        y_parts = []
        j_parts = []
        for traj in self.trajectories:
            y_parts.append((traj.A @ p + traj.b).reshape(-1))
            j_parts.append(traj.A.copy())
        return np.concatenate(y_parts, axis=0), [np.vstack(j_parts)]


@dataclass
class TimeDiffExpr:
    """First-order backward difference over time-stacked vectors.

    For y = [y(0), y(1), ..., y(T-1)] with each y(k) in R^segment_dim,
    returns [y(1)-y(0), ..., y(T-1)-y(T-2)].
    """

    name: str
    base: Expr
    segment_dim: int
    scale: float = 1.0
    use_time_dt: bool = False
    dt: float | None = None

    @property
    def vars(self):
        return self.base.vars

    def deps(self):
        return self.base.deps()

    def value_deps(self):
        deps = getattr(self.base, "value_deps", None)
        if not callable(deps):
            raise AttributeError(f"{self.name}: value_deps fast path is not available.")
        return deps()

    def eval_value(self, ctx: RuntimeContext):
        value = getattr(self.base, "eval_value", None)
        if not callable(value):
            raise AttributeError(f"{self.name}: value fast path is not available.")
        y = np.asarray(value(ctx), dtype=float).reshape(-1)

        seg = int(self.segment_dim)
        if seg <= 0:
            raise ValueError(f"{self.name}: segment_dim must be > 0, got {seg}.")
        if y.size % seg != 0:
            raise ValueError(
                f"{self.name}: base size {y.size} is not divisible by segment_dim={seg}."
            )

        steps = int(y.size // seg)
        if steps < 2:
            raise ValueError(f"{self.name}: need at least 2 steps, got {steps}.")

        y2 = y.reshape(steps, seg)

        scale = float(self.scale)
        if self.use_time_dt:
            dt = self.dt
            if dt is None:
                time = getattr(ctx, "time", None)
                if time is None or not hasattr(time, "dt"):
                    raise ValueError(f"{self.name}: wrt='time' requires time.dt in context or explicit dt in DSL.")
                try:
                    dt = float(time.dt)
                except Exception as e:
                    raise ValueError(f"{self.name}: invalid time.dt in context: {getattr(time, 'dt', None)!r}") from e
            if dt <= 0.0:
                raise ValueError(f"{self.name}: dt must be > 0 for wrt='time', got {dt}.")
            scale = scale / float(dt)

        return scale * (y2[1:, :] - y2[:-1, :]).reshape(-1)

    def eval(self, ctx: RuntimeContext):
        y, blocks = self.base.eval(ctx)
        y = np.asarray(y, dtype=float).reshape(-1)

        seg = int(self.segment_dim)
        if seg <= 0:
            raise ValueError(f"{self.name}: segment_dim must be > 0, got {seg}.")
        if y.size % seg != 0:
            raise ValueError(
                f"{self.name}: base size {y.size} is not divisible by segment_dim={seg}."
            )

        steps = int(y.size // seg)
        if steps < 2:
            raise ValueError(f"{self.name}: need at least 2 steps, got {steps}.")

        y2 = y.reshape(steps, seg)

        scale = float(self.scale)
        if self.use_time_dt:
            dt = self.dt
            if dt is None:
                time = getattr(ctx, "time", None)
                if time is None or not hasattr(time, "dt"):
                    raise ValueError(f"{self.name}: wrt='time' requires time.dt in context or explicit dt in DSL.")
                try:
                    dt = float(time.dt)
                except Exception as e:
                    raise ValueError(f"{self.name}: invalid time.dt in context: {getattr(time, 'dt', None)!r}") from e
            if dt <= 0.0:
                raise ValueError(f"{self.name}: dt must be > 0 for wrt='time', got {dt}.")
            scale = scale / float(dt)

        r = scale * (y2[1:, :] - y2[:-1, :]).reshape(-1)

        blocks2 = []
        for B in blocks:
            Bm = np.asarray(B, dtype=float)
            if Bm.ndim != 2 or Bm.shape[0] != y.size:
                raise ValueError(
                    f"{self.name}: block row mismatch. base size={y.size}, block shape={Bm.shape}."
                )
            B3 = Bm.reshape(steps, seg, Bm.shape[1])
            Bd = scale * (B3[1:, :, :] - B3[:-1, :, :]).reshape((steps - 1) * seg, Bm.shape[1])
            blocks2.append(Bd)
        return r, blocks2


@dataclass
class ConstantExpr:
    name: str
    value: np.ndarray
    vars: Sequence[Variable] = ()

    def deps(self):
        return []

    def eval_value(self, ctx: RuntimeContext):
        del ctx
        return np.asarray(self.value, dtype=float).reshape(-1)

    def eval(self, ctx: RuntimeContext):
        y = self.eval_value(ctx)
        blocks = [np.zeros((y.size, v.dim()), dtype=float) for v in self.vars]
        return y, blocks

    def vjp(self, ctx: RuntimeContext, rhs):
        del ctx
        r = np.asarray(rhs, dtype=float)
        return [
            np.zeros((v.dim(),), dtype=float) if r.ndim == 1 else np.zeros((v.dim(), int(r.shape[1])), dtype=float)
            for v in self.vars
        ]


@dataclass
class RepeatConstantExpr:
    """Repeat a constant segment vector along the time axis."""

    name: str
    value: np.ndarray
    repeats: int
    vars: Sequence[Variable] = ()

    def deps(self):
        return []

    def eval_value(self, ctx: RuntimeContext):
        del ctx
        base = np.asarray(self.value, dtype=float).reshape(-1)
        repeats = int(self.repeats)
        if repeats <= 0:
            raise ValueError(f"{self.name}: repeats must be > 0, got {repeats}.")
        return np.tile(base, repeats)

    def eval(self, ctx: RuntimeContext):
        y = self.eval_value(ctx)
        blocks = [np.zeros((y.size, v.dim()), dtype=float) for v in self.vars]
        return y, blocks

    def vjp(self, ctx: RuntimeContext, rhs):
        del ctx
        r = np.asarray(rhs, dtype=float)
        return [
            np.zeros((v.dim(),), dtype=float) if r.ndim == 1 else np.zeros((v.dim(), int(r.shape[1])), dtype=float)
            for v in self.vars
        ]


@dataclass
class SubExpr:
    name: str
    a: Expr
    b: Expr

    @property
    def vars(self):
        return self.a.vars

    def deps(self):
        return list(self.a.deps()) + list(self.b.deps())

    def value_deps(self):
        deps_a = getattr(self.a, "value_deps", None)
        deps_b = getattr(self.b, "value_deps", None)
        if not callable(deps_a) or not callable(deps_b):
            raise AttributeError(f"{self.name}: value_deps fast path is not available.")
        return list(deps_a()) + list(deps_b())

    def eval_value(self, ctx: RuntimeContext):
        value_a = getattr(self.a, "eval_value", None)
        value_b = getattr(self.b, "eval_value", None)
        if not callable(value_a) or not callable(value_b):
            raise AttributeError(f"{self.name}: value fast path is not available.")
        ra = np.asarray(value_a(ctx), dtype=float).reshape(-1)
        rb = np.asarray(value_b(ctx), dtype=float).reshape(-1)
        if ra.shape != rb.shape:
            raise ValueError(f"{self.name}: shape mismatch {ra.shape} vs {rb.shape}")
        return ra - rb

    def eval(self, ctx: RuntimeContext):
        ra, Ja = self.a.eval(ctx)
        rb, Jb = self.b.eval(ctx)
        if ra.shape != rb.shape:
            raise ValueError(f"{self.name}: shape mismatch {ra.shape} vs {rb.shape}")
        if len(Ja) != len(Jb):
            raise ValueError(f"{self.name}: block len mismatch {len(Ja)} vs {len(Jb)}")
        r = ra - rb
        blocks = [A - B for A, B in zip(Ja, Jb)]
        return r, blocks

    def vjp(self, ctx: RuntimeContext, rhs):
        r = np.asarray(rhs, dtype=float)
        vjp_a = getattr(self.a, "vjp", None)
        vjp_b = getattr(self.b, "vjp", None)
        if not callable(vjp_a) or not callable(vjp_b):
            raise AttributeError(f"{self.name}: vjp fast path is not available.")
        ga = [np.asarray(g, dtype=float) for g in vjp_a(ctx, r)]
        gb = [np.asarray(g, dtype=float) for g in vjp_b(ctx, r)]
        if len(ga) != len(gb):
            raise ValueError(f"{self.name}: vjp block len mismatch {len(ga)} vs {len(gb)}")
        return [a - b for a, b in zip(ga, gb)]


@dataclass
class StackExpr:
    name: str
    parts: Sequence[Expr]

    @property
    def vars(self):
        return self.parts[0].vars if self.parts else []

    def deps(self):
        out = []
        for p in self.parts:
            out.extend(list(p.deps()))
        return out

    def value_deps(self):
        out = []
        for p in self.parts:
            deps = getattr(p, "value_deps", None)
            if not callable(deps):
                raise AttributeError(f"{self.name}: value_deps fast path is not available.")
            out.extend(list(deps()))
        return out

    def eval_value(self, ctx: RuntimeContext):
        r_list = []
        for p in self.parts:
            value = getattr(p, "eval_value", None)
            if not callable(value):
                raise AttributeError(f"{self.name}: value fast path is not available.")
            r_list.append(np.asarray(value(ctx), float).reshape(-1))
        return np.concatenate(r_list, axis=0) if r_list else np.zeros((0,), float)

    def eval(self, ctx: RuntimeContext):
        r_list = []
        J_list = None
        for p in self.parts:
            r, blocks = p.eval(ctx)
            r_list.append(np.asarray(r, float).reshape(-1))
            if J_list is None:
                J_list = [[] for _ in blocks]
            for i, B in enumerate(blocks):
                J_list[i].append(np.asarray(B, float))
        r_all = np.concatenate(r_list, axis=0) if r_list else np.zeros((0,), float)
        blocks_all = [np.vstack(chunks) for chunks in (J_list or [])]
        return r_all, blocks_all


@dataclass
class ComponentExpr:
    """Select one component from vectors stacked as (..., segment_dim)."""

    name: str
    base: Expr
    segment_dim: int
    index: int

    @property
    def vars(self):
        return self.base.vars

    def deps(self):
        return self.base.deps()

    def value_deps(self):
        deps = getattr(self.base, "value_deps", None)
        if not callable(deps):
            raise AttributeError(f"{self.name}: value_deps fast path is not available.")
        return deps()

    def eval_value(self, ctx: RuntimeContext):
        value = getattr(self.base, "eval_value", None)
        if not callable(value):
            raise AttributeError(f"{self.name}: value fast path is not available.")
        y = np.asarray(value(ctx), dtype=float).reshape(-1)

        seg = int(self.segment_dim)
        idx = int(self.index)
        if seg <= 0:
            raise ValueError(f"{self.name}: segment_dim must be > 0, got {seg}.")
        if idx < 0 or idx >= seg:
            raise ValueError(f"{self.name}: index must be in [0, {seg - 1}], got {idx}.")
        if y.size % seg != 0:
            raise ValueError(
                f"{self.name}: base size {y.size} is not divisible by segment_dim={seg}."
            )

        steps = int(y.size // seg)
        return y.reshape(steps, seg)[:, idx].reshape(-1)

    def eval(self, ctx: RuntimeContext):
        y, blocks = self.base.eval(ctx)
        y = np.asarray(y, dtype=float).reshape(-1)

        seg = int(self.segment_dim)
        idx = int(self.index)
        if seg <= 0:
            raise ValueError(f"{self.name}: segment_dim must be > 0, got {seg}.")
        if idx < 0 or idx >= seg:
            raise ValueError(f"{self.name}: index must be in [0, {seg - 1}], got {idx}.")
        if y.size % seg != 0:
            raise ValueError(
                f"{self.name}: base size {y.size} is not divisible by segment_dim={seg}."
            )

        steps = int(y.size // seg)
        y2 = y.reshape(steps, seg)[:, idx].reshape(-1)

        blocks2 = []
        for B in blocks:
            Bm = np.asarray(B, dtype=float)
            if Bm.ndim != 2 or Bm.shape[0] != y.size:
                raise ValueError(
                    f"{self.name}: block row mismatch. base size={y.size}, block shape={Bm.shape}."
                )
            blocks2.append(Bm.reshape(steps, seg, Bm.shape[1])[:, idx, :].reshape(steps, Bm.shape[1]))
        return y2, blocks2


@dataclass
class HingeExpr:
    name: str
    base: Expr

    @property
    def vars(self):
        return self.base.vars

    def deps(self):
        return self.base.deps()

    def value_deps(self):
        deps = getattr(self.base, "value_deps", None)
        if not callable(deps):
            raise AttributeError(f"{self.name}: value_deps fast path is not available.")
        return deps()

    def eval_value(self, ctx: RuntimeContext):
        value = getattr(self.base, "eval_value", None)
        if not callable(value):
            raise AttributeError(f"{self.name}: value fast path is not available.")
        h = np.asarray(value(ctx), dtype=float).reshape(-1)
        return np.maximum(0.0, h)

    def eval(self, ctx: RuntimeContext):
        h, blocks = self.base.eval(ctx)

        h = np.asarray(h, dtype=float).reshape(-1)
        m = h.size

        active = (h > 0.0).astype(float)
        r = np.maximum(0.0, h)

        blocks2 = []
        for B in blocks:
            B = np.asarray(B, dtype=float)
            if B.shape[0] != m:
                raise ValueError(f"{self.name}: block row mismatch: h has {m}, block has {B.shape}")
            blocks2.append(active[:, None] * B)
        return r, blocks2
