"""Curvature of provably affine residual terms, excluding state derivatives."""
from __future__ import annotations

import numpy as np

from ..core.expr.nodes import (
    ComponentExpr, ConstantExpr, GetVarExpr, RepeatConstantExpr, StackExpr,
    SubExpr, TimeDiffExpr, TrajectoryVarDerivativesExpr, TrajectoryVarExpr,
)
from .costs import DiagonalWeightCost, L2Cost, ScalarWeightCost


def _affine(expr):
    # Exact types: custom subclasses may change the mathematical operation.
    if type(expr) in (ConstantExpr, RepeatConstantExpr, GetVarExpr,
                      TrajectoryVarExpr, TrajectoryVarDerivativesExpr):
        return True
    if type(expr) in (ComponentExpr, TimeDiffExpr):
        return _affine(expr.base)
    if type(expr) is SubExpr:
        return _affine(expr.a) and _affine(expr.b)
    if type(expr) is StackExpr:
        return all(_affine(part) for part in expr.parts)
    return False


def linear_residual_gram(runtime, *, weighted=True, term_indices=None, max_size=512):
    """Return sum J_linear.T J_linear, or None if absent/over size budget.

    Only cheap affine expressions and constant built-in residual weights are
    admitted. No state update, dynamics derivative, random probing, or global
    nonlinear Jacobian is performed. Temporary blocks of individual affine
    terms can be dense. The size cap bounds the dense preconditioner itself.
    """
    n = runtime.pack.n_total
    if n > max_size:
        return None
    gram = np.zeros((n, n))
    found = False
    for idx in runtime._normalize_term_indices(term_indices):
        expr, cost = runtime.problem.terms[idx]
        if not _affine(expr) or (weighted and type(cost) not in
                                (L2Cost, ScalarWeightCost, DiagonalWeightCost)):
            continue
        r, blocks = expr.eval(runtime.ctx)
        if weighted:
            r, blocks = cost.apply(r, blocks)
        # Align local blocks once; never assemble the complete residual stack.
        columns = {}
        for var, block in zip(expr.vars, blocks, strict=True):
            start, stop = runtime.pack.slices[var.name]
            block = np.asarray(block, dtype=float)
            if block.shape != (np.asarray(r).size, stop-start):
                raise ValueError("Linear preconditioner: invalid derivative block.")
            if var.name in columns:
                columns[var.name] += block
            else:
                columns[var.name] = block.copy()
        for name_a, a in columns.items():
            sa, ea = runtime.pack.slices[name_a]
            for name_b, b in columns.items():
                sb, eb = runtime.pack.slices[name_b]
                gram[sa:ea, sb:eb] += a.T @ b
        found = True
    return gram if found else None
