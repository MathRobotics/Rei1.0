"""Point-bound Jacobian products and a NumPy-only damped CGLS solve."""
from __future__ import annotations

import numpy as np

from ...problem import NLSRuntimeLinearProblem, as_linearized_problem


def _vector(value, size, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name}: expected a finite vector of shape {(size,)}, got {value.shape}.")
    return value


class JacobianProducts:
    """Bind products to a point even while line search mutates the model."""

    def __init__(self, model, required, rows):
        self.model, self.required = model, required
        self.point = np.asarray(model.get_point(), dtype=float).copy()
        self.shape = (rows, int(model.n_total))
        self._diagonal = None

    def _at_point(self, fn):
        previous = np.asarray(self.model.get_point(), dtype=float).copy()
        changed = not np.array_equal(previous, self.point)
        try:
            if changed:
                self.model.set_point(self.point)
            return fn()
        finally:
            if changed:
                self.model.set_point(previous)

    def __matmul__(self, v):
        v = _vector(v, self.shape[1], "jvp input")
        return _vector(self._at_point(lambda: self.model.jvp(v, required=self.required)),
                       self.shape[0], "jvp output")

    def transpose_product(self, w):
        w = _vector(w, self.shape[0], "vjp input")
        return _vector(self._at_point(lambda: self.model.vjp(w, required=self.required)),
                       self.shape[1], "vjp output")

    @property
    def T(self):
        return _TransposeProducts(self)

    def column_squared_norms(self):
        if self._diagonal is None:
            fn = getattr(self.model, "jacobian_column_squared_norms", None)
            if callable(fn):
                diagonal = _vector(self._at_point(lambda: fn(required=self.required)),
                                   self.shape[1], "Jacobian column squared norms")
                if np.any(diagonal < 0):
                    raise ValueError("Jacobian column squared norms must be nonnegative.")
            else:
                diagonal = np.empty(self.shape[1])
                basis = np.zeros(self.shape[1])
                for i in range(self.shape[1]):
                    basis[i] = 1.
                    column = self @ basis
                    diagonal[i] = column @ column
                    basis[i] = 0.
            self._diagonal = diagonal.copy()
        return self._diagonal


class _TransposeProducts:
    def __init__(self, operator):
        self.operator = operator

    def __matmul__(self, w):
        return self.operator.transpose_product(w)


class OperatorLinearizationProblem:
    """Use runtime selection adapters, or a generic eval/jvp/vjp provider."""

    def __init__(self, problem, *, weighted=None, term_indices=None):
        if isinstance(problem, NLSRuntimeLinearProblem) or (
            hasattr(problem, "pack") and hasattr(problem, "linearize_stacked_terms")
        ):
            model = as_linearized_problem(problem, weighted=weighted, term_indices=term_indices)
        else:
            if weighted is False or term_indices is not None:
                raise ValueError("Operator problem: weight/term selection requires an NLS runtime.")
            model = problem
        for name in ("get_point", "set_point", "required_list", "eval", "jvp", "vjp"):
            if not callable(getattr(model, name, None)):
                raise TypeError(f"Operator problem requires {name}(...).")
        self.n_total = int(model.n_total)
        self.model = model

    def get_point(self):
        return self.model.get_point()

    def set_point(self, x):
        self.model.set_point(x)

    def required_list(self, required=None):
        fn = getattr(self.model, "operator_required_list", None)
        if callable(fn):
            return fn(required)
        return self.model.required_list(required)

    def eval(self, *, required=None):
        return self.model.eval(required=required)

    def linearize(self, *, required=None):
        req = None if required is None else tuple(required)
        r = np.asarray(self.eval(required=req), dtype=float).reshape(-1)
        _vector(r, r.size, "residual")
        _vector(self.get_point(), self.n_total, "point")
        return r, JacobianProducts(self.model, req, r.size)


def cgls_step(J, r, damping, *, tolerance, max_iters):
    """Solve min ||J h+r||²+lambda||h||² with augmented CGLS.

    Keep the residual of both augmented blocks, avoiding J.T J formation.
    Verify the true normal residual before declaring inner convergence.
    """
    n = J.shape[1]
    limit = max(20, 2 * n) if max_iters is None else int(max_iters)
    root = np.sqrt(damping)
    h = np.zeros(n)
    data_residual = -np.asarray(r, dtype=float).copy()
    regularization_residual = np.zeros(n)
    s = J.T @ data_residual
    initial = float(np.linalg.norm(s))
    threshold = tolerance * initial
    p = s.copy()
    gamma = float(s @ s)
    status = "converged" if initial == 0 else "max_iters"
    iteration = 0
    for iteration in range(1, limit + 1):
        if initial == 0:
            iteration = 0
            break
        q = J @ p
        t = root * p
        denominator = float(q @ q + t @ t)
        if not np.isfinite(denominator) or denominator <= 0 or not np.isfinite(gamma):
            raise ValueError("CGLS breakdown: non-finite or nonpositive curvature; rescale the problem.")
        alpha = gamma / denominator
        h += alpha * p
        data_residual -= alpha * q
        regularization_residual -= alpha * t
        s = J.T @ data_residual + root * regularization_residual
        new_gamma = float(s @ s)
        if np.sqrt(new_gamma) <= threshold or iteration == limit:
            data_residual = -(r + J @ h)
            regularization_residual = -root * h
            s = J.T @ data_residual + root * regularization_residual
            new_gamma = float(s @ s)
            if np.sqrt(new_gamma) <= threshold:
                status = "converged"
                break
            # Restart after replacing the recursively updated residual.
            p = s.copy()
        else:
            p = s + (new_gamma / gamma) * p
        gamma = new_gamma
    _vector(h, n, "CGLS step")
    _vector(s, n, "CGLS normal residual")
    norm = float(np.linalg.norm(s))
    return h, dict(iterations=iteration, status=status, normal_residual_norm=norm,
                   relative_normal_residual=norm / initial if initial else 0.,
                   tolerance=tolerance, max_iters=limit)
