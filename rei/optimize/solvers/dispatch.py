"""Resolve solver options and dispatch to the selected implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from operator import index
from typing import Any

import numpy as np

from ...core.state_cache import StateKey
from ...core.timing import Profiler
from ._external_common import IterCallback, SolveResult, _format_option_names
from .cyipopt import solve_cyipopt_minimize
from .gauss_newton import solve_gauss_newton
from .gauss_newton_operator import solve_gauss_newton_operator
from .gauss_newton_krylov import solve_gauss_newton_krylov
from .levenberg_marquardt import solve_levenberg_marquardt
from .liteopt import solve_liteopt_gd
from .lm_ls import solve_lm_ls
from .nls import nls
from .scipy import solve_scipy_minimize

Array = np.ndarray

_COMMON_SOLVE_OPTION_KEYS = frozenset(
    {
        "x0",
        "weighted",
        "term_indices",
    }
)

_SOLVER_REI_OPTION_KEYS: dict[str, frozenset[str]] = {
    "lm-ls": frozenset({
        "max_iters", "tol_grad", "tau", "damping", "damping_min_factor",
        "damping_increase", "damping_decrease", "damping_max", "ls_beta",
        "ls_min_step", "ls_max_iters", "ls_max_retries", "c_armijo",
        "history", "history_vectors", "history_path", "line_search_history_path", "verbose",
    }),
    "levenberg_marquardt": frozenset({
        "max_iters", "tol_grad", "tol_dx", "tau", "damping",
        "history", "history_vectors", "history_path", "trial_history_path", "verbose",
    }),
    "gauss_newton": frozenset(
        {
            "max_iters",
            "tol_r",
            "tol_dx",
            "tol_grad",
            "damping",
            "line_search",
            "ls_beta",
            "ls_min_step",
            "ls_max_iters",
            "ls_max_retries",
            "damping_increase",
            "damping_max",
            "damping_decrease",
            "damping_min_factor",
            "c_armijo",
            "history",
            "history_vectors",
            "history_path",
            "line_search_history_path",
            "verbose",
        }
    ),
    "scipy_minimize": frozenset(
        {
            "method",
            "max_iters",
            "tol",
            "bounds",
            "verbose",
        }
    ),
    "cyipopt": frozenset(
        {
            "max_iters",
            "tol",
            "bounds",
            "verbose",
        }
    ),
    "liteopt": frozenset(
        {
            "method",
            "verbose",
            "max_iters",
            "step_size",
            "tol_grad",
            "tol_r",
            "tol_dx",
            "lambda_",
            "step_scale",
            "damping_update",
            "linear_system",
            "line_search_method",
            "line_search",
            "ls_beta",
            "ls_min_step",
            "ls_max_steps",
            "ls_max_iters",
            "c_armijo",
            "nonfinite_retries",
            "nonfinite_step_shrink",
            "min_step_size",
        }
    ),
}

_SOLVER_REI_OPTION_KEYS["gauss_newton_operator"] = (
    _SOLVER_REI_OPTION_KEYS["gauss_newton"] | {"inner_tol", "inner_max_iters"}
)

_SOLVER_REI_OPTION_KEYS["gauss_newton_krylov"] = frozenset(
    _SOLVER_REI_OPTION_KEYS["gauss_newton"] | {
        "inner_tol", "inner_max_iters", "globalization", "initial_radius", "max_radius", "acceptance",
        "forcing_min", "forcing_max", "preconditioner",
        "preconditioner_probes", "preconditioner_max_size", "preconditioner_floor",
        "preconditioner_refresh", "seed", "history", "history_vectors", "history_path",
        "line_search_history_path", "trial_history_path", "verbose",
    }
)

_BUILTIN_NLS_SOLVERS = frozenset({
    "gauss_newton", "gauss_newton_operator", "gauss_newton_krylov",
    "levenberg_marquardt", "lm-ls",
})

_ALL_REI_SOLVER_OPTION_KEYS = frozenset().union(*_SOLVER_REI_OPTION_KEYS.values())

# Only defaults that differ from the direct solver APIs belong here.
_DISPATCH_DEFAULTS = {
    "gauss_newton": {"max_iters": 200},
    "gauss_newton_operator": {"max_iters": 200},
}
_LITEOPT_GD_ONLY_OPTIONS = frozenset({
    "step_size", "tol_grad", "nonfinite_retries", "nonfinite_step_shrink", "min_step_size",
})
_LITEOPT_GN_ONLY_OPTIONS = _SOLVER_REI_OPTION_KEYS["liteopt"] - (
    _LITEOPT_GD_ONLY_OPTIONS | {"method", "verbose", "max_iters", "line_search"}
)
_LITEOPT_GN_DEFAULTS = {
    "tol_r": 1e-10,
    "tol_dx": 1e-12,
    "lambda_": 1e-8,
    "damping_update": "fixed",
    "linear_system": "normal_jtj",
    "line_search_method": "strict_decrease",
    "line_search": True,
    "ls_beta": 0.5,
    "ls_min_step": 1e-8,
    "ls_max_steps": 12,
}
_INTEGER_OPTION_MINIMUMS = {
    "max_iters": 0,
    "ls_max_iters": 1,
    "ls_max_steps": 1,
    "ls_max_retries": 0,
    "inner_max_iters": 1,
    "nonfinite_retries": 0,
}


@dataclass(frozen=True)
class _SolveRequest:
    solver: str
    arguments: dict[str, Any]


def _allowed_solve_option_keys(solver_key: str) -> frozenset[str]:
    allowed = set(_COMMON_SOLVE_OPTION_KEYS)
    allowed.update(_SOLVER_REI_OPTION_KEYS[solver_key])
    if solver_key not in _BUILTIN_NLS_SOLVERS:
        allowed.add("backend_options")
    return frozenset(allowed)


def _normalize_backend_options_for_solver(
    options: Mapping[str, Any],
    *,
    solver_key: str,
) -> Mapping[str, Any] | None:
    allowed = _allowed_solve_option_keys(solver_key)
    unknown = tuple(k for k in options if k not in allowed)
    if not unknown:
        if solver_key in _BUILTIN_NLS_SOLVERS:
            return None
        return _as_options_mapping(
            options.get("backend_options", None),
            where=f"solve({solver_key})",
        )

    if solver_key in _BUILTIN_NLS_SOLVERS:
        allowed_text = _format_option_names(_allowed_solve_option_keys(solver_key))
        raise ValueError(
            f"solve({solver_key}): unsupported option(s): "
            f"{_format_option_names(unknown)}. "
            f"Allowed options are: {allowed_text}."
        )

    foreign_rei = tuple(
        k for k in unknown if str(k) in (_ALL_REI_SOLVER_OPTION_KEYS - _SOLVER_REI_OPTION_KEYS[solver_key])
    )
    if foreign_rei:
        allowed_text = _format_option_names(_allowed_solve_option_keys(solver_key))
        raise ValueError(
            f"solve({solver_key}): unsupported solver option(s): "
            f"{_format_option_names(foreign_rei)}. "
            f"Allowed options are: {allowed_text}. "
            "Backend-specific options must be under options['backend_options'] "
            "or additional top-level keys that are not rei solver options."
        )

    backend = _as_options_mapping(
        options.get("backend_options", None),
        where=f"solve({solver_key})",
    )
    backend_local: dict[str, Any] = {} if backend is None else dict(backend)
    for k in unknown:
        if k not in backend_local:
            backend_local[k] = options[k]
    return backend_local


def _merge_options(*sources: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for src in sources:
        if src is None:
            continue
        if not isinstance(src, Mapping):
            raise TypeError("solve: options must be a mapping or None.")
        if any(not isinstance(key, str) for key in src):
            raise TypeError("solve: option names must be strings.")
        merged.update(dict(src))
    return merged


def _as_options_mapping(value: Any, *, where: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{where}: options['backend_options'] must be a mapping or None.")
    return dict(value)


def _normalize_term_indices_option(raw: Any) -> tuple[int, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)):
        raise TypeError("term_indices must be an iterable of integers, not a string.")
    try:
        vals = []
        for value in raw:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError("boolean term index")
            vals.append(index(value))
    except TypeError as e:
        raise TypeError("term_indices must be an iterable of integers or None.") from e
    return tuple(vals)


def _validate_integer_options(solver: str, options: Mapping[str, Any]) -> None:
    """Reject fractional counts before a backend can truncate them or set x0."""
    for name, minimum in _INTEGER_OPTION_MINIMUMS.items():
        if name not in options:
            continue
        value = options[name]
        if value is None and (
            name == "inner_max_iters"
            or (name == "max_iters" and solver not in _BUILTIN_NLS_SOLVERS)
        ):
            continue
        message = f"solve({solver}): {name} must be a finite integer >= {minimum}."
        try:
            valid = (
                not isinstance(value, (bool, np.bool_, str, bytes))
                and np.isscalar(value)
                and np.isfinite(value)
                and int(value) == value
                and value >= minimum
            )
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(message)


def _prepare_solver_arguments(
    solver: str, options: Mapping[str, Any], backend_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply dispatch defaults and translate only the selected solver's options."""
    arguments = dict(_DISPATCH_DEFAULTS.get(solver, {}))
    arguments.update({
        key: value for key, value in options.items() if key in _SOLVER_REI_OPTION_KEYS[solver]
    })
    if solver == "liteopt":
        method = str(arguments.get("method", "gd")).strip().lower()
        if method not in {"gd", "gn"}:
            raise ValueError(
                "solve_liteopt_gd: liteopt method must be 'gd' or 'gn'. "
                f"Got method={options.get('method')!r}."
            )
        incompatible = _LITEOPT_GN_ONLY_OPTIONS if method == "gd" else _LITEOPT_GD_ONLY_OPTIONS
        invalid = incompatible.intersection(arguments)
        if invalid:
            raise ValueError(
                f"solve(liteopt): unsupported option(s) for method={method!r}: "
                f"{_format_option_names(invalid)}."
            )
        arguments["method"] = method
        if method == "gn":
            # Keep ls_max_steps precedence over the compatibility spelling.
            if "ls_max_iters" in arguments:
                arguments.setdefault("ls_max_steps", arguments.pop("ls_max_iters"))
            arguments = {**_LITEOPT_GN_DEFAULTS, **arguments}
            if arguments["line_search"] is None:
                arguments["line_search"] = True
    _validate_integer_options(solver, arguments)
    if solver not in _BUILTIN_NLS_SOLVERS:
        arguments["options"] = backend_options
    return arguments


def _resolve_solve_request(
    problem: Any, *, solver: str | None, x0: Any, options: Mapping[str, Any] | None,
) -> _SolveRequest:
    """Resolve configuration and validate arguments without evaluating the problem."""
    from ..solver_config import normalize_solver_config

    raw_config = getattr(problem, "solver_config", None)
    config = normalize_solver_config(raw_config) if raw_config else {}
    selected = solver if solver is not None else config.get("name", "levenberg_marquardt")
    key = str(selected).strip().lower()
    if key not in _SOLVER_REI_OPTION_KEYS:
        raise ValueError(
            "Unknown solver. Use one of: "
            f"{', '.join(repr(name) for name in _SOLVER_REI_OPTION_KEYS)}. "
            "Solver aliases are not supported. "
            f"Got solver={selected!r}."
        )
    # A different explicit solver starts with its own defaults.
    configured_options = config.get("options") if key == config.get("name") else None
    opts = _merge_options(configured_options, options)
    if x0 is not None and "x0" in opts:
        raise ValueError("solve: pass x0 either as keyword argument or options['x0'], not both.")
    weighted = opts.get("weighted")
    if weighted is not None and not isinstance(weighted, (bool, np.bool_)):
        raise TypeError("solve: weighted must be a boolean or None.")
    term_indices = _normalize_term_indices_option(opts.get("term_indices"))
    backend_options = _normalize_backend_options_for_solver(opts, solver_key=key)
    arguments = _prepare_solver_arguments(key, opts, backend_options)
    arguments.update(
        x0=opts.get("x0", x0),
        weighted=None if weighted is None else bool(weighted),
        term_indices=term_indices,
    )
    return _SolveRequest(solver=key, arguments=arguments)


def _select_solver(solver: str) -> Callable[..., SolveResult]:
    """Select the implementation independently of option resolution."""
    return {
        "levenberg_marquardt": solve_levenberg_marquardt,
        "lm-ls": solve_lm_ls,
        "gauss_newton": solve_gauss_newton,
        "gauss_newton_operator": solve_gauss_newton_operator,
        "gauss_newton_krylov": solve_gauss_newton_krylov,
        "scipy_minimize": solve_scipy_minimize,
        "cyipopt": solve_cyipopt_minimize,
        "liteopt": solve_liteopt_gd,
    }[solver]


def solve(
    problem: Any,
    *,
    solver: str | None = None,
    x0: Array | Any = None,
    required: Iterable[StateKey] | None = None,
    on_iter: IterCallback | None = None,
    options: Mapping[str, Any] | None = None,
    profiler: Profiler | None = None,
) -> SolveResult:
    """Dispatch runtime solve (default: classical Levenberg-Marquardt).

    Returns:
      SolveOutcome(solution, stats, timing, meta)

    Compiled TOML/DSL solver defaults are used when solver is omitted.
    Explicit options override configured options by key. Selecting a different
    solver discards configured options. With no configuration, the default is LM.
    Solver parameters are provided via `options`.
    `x0` may be passed directly or as `options["x0"]` (but not both).
    Common arguments and integer counts are checked before invoking a solver.
    `weighted` must be bool or None, and term indices must be integers.
    Liteopt GD/GN reject options that apply only to the other method.

    levenberg_marquardt (default):
      max_iters=200, tol_grad=1e-8, tol_dx=1e-12, tau=1e-3, damping=None
      history=True, history_vectors=False, history_path=None, trial_history_path=None
      verbose=True; no line search or custom roundoff acceptance

    lm-ls:
      LM directions with Armijo backtracking; tol_grad=1e-8 is the only
      convergence criterion. max_iters counts accepted updates.

    gauss_newton (previous adaptive method):
      max_iters, tol_r, tol_dx, tol_grad, damping, line_search, ls_beta, ls_min_step, ls_max_iters
      ls_max_retries=3, damping_increase=10.0, damping_max=1e12
      damping_decrease=0.1, c_armijo=1e-4
      history=True, history_vectors=False, history_path=None, line_search_history_path=None
      verbose=True (print state history during the solve)

    gauss_newton_operator:
      gauss_newton options plus inner_tol=1e-8, inner_max_iters=None
      Uses eval/jvp/vjp and damped CGLS; runtime local-block fallback supported.

    gauss_newton_krylov:
      Same adaptive damping and Armijo line-search algorithm as gauss_newton;
      the damped least-squares step uses preconditioned CGLS with JVP/VJP.
      inner_tol=1e-10 by default; None opts into adaptive accuracy and an initial
      50-step budget that grows on limit exits. Numeric budgets are fixed caps.
      globalization="trust_region" selects the legacy PCG trust-region method.

    scipy_minimize:
      method, max_iters, tol, bounds, backend_options
      (unknown top-level keys are forwarded to scipy's options dict)

    cyipopt:
      max_iters, tol, bounds, backend_options
      (unknown top-level keys are forwarded to IPOPT options dict)

    liteopt:
      method='gd' (default):
        max_iters, step_size, tol_grad, line_search, verbose,
        nonfinite_retries, nonfinite_step_shrink, min_step_size
      method='gn':
        max_iters, tol_r, tol_dx, lambda_, step_scale,
        damping_update, linear_system,
        line_search_method, line_search, ls_beta, ls_min_step, ls_max_steps,
        verbose
      backend_options
      (unknown top-level keys are forwarded to the selected liteopt backend API)
    """

    if on_iter is not None and not callable(on_iter):
        raise TypeError("solve: on_iter must be callable or None.")
    request = _resolve_solve_request(problem, solver=solver, x0=x0, options=options)
    implementation = _select_solver(request.solver)
    return implementation(
        problem, required=required, on_iter=on_iter, profiler=profiler, **request.arguments,
    )


__all__ = [
    "nls",
    "solve",
    "solve_gauss_newton",
    "solve_gauss_newton_operator",
    "solve_gauss_newton_krylov",
    "solve_levenberg_marquardt",
    "solve_lm_ls",
    "solve_scipy_minimize",
    "solve_cyipopt_minimize",
    "solve_liteopt_gd",
]
