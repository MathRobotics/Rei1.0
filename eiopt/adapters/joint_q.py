from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import numpy as np

from ..core.state_cache import OwnerKey, StateKey
from ..core.state_schema import DEFAULT_ROBOT_NAME, DTYPE_JOINT, jac_field

Array = np.ndarray


def _call_build_state(
    build_state: Callable[..., dict],
    x_all: Array,
    *,
    pack: Any = None,
    time: Any = None,
    required: Optional[Iterable[StateKey]] = None,
) -> dict:
    try:
        return build_state(x_all, pack=pack, time=time, required=required)
    except TypeError:
        try:
            return build_state(x_all, time=time, required=required)
        except TypeError:
            try:
                return build_state(x_all, time=time)
            except TypeError:
                return build_state(x_all)


def with_standard_joint_q(
    build_state: Callable[..., dict],
    *,
    q_var: str = "q",
    owner_type: str = "total_joint",
    owner_name: str = DEFAULT_ROBOT_NAME,
    dtype: str = DTYPE_JOINT,
    field: str = "q",
) -> Callable[..., dict]:
    """Wrap backend `build_state()` and inject standard joint-angle keys.

    Injected keys (only when requested via `required`):
      - StateKey(..., dtype="joint", field="q")      -> q(k)
      - StateKey(..., dtype="joint", field="q_J_q")  -> dq(k)/dq_all

    Time indexing:
      - If `time` is provided and q-dimension is divisible by (time.N+1), we treat q as
        a stacked trajectory and slice by `k`.
      - Otherwise only k=0 is supported.
    """

    owner = OwnerKey(owner_type=str(owner_type), owner_name=str(owner_name))
    jac_name = jac_field(field, var=q_var)

    def wrapped(
        x_all: Array,
        *,
        pack: Any = None,
        time: Any = None,
        required: Optional[Iterable[StateKey]] = None,
    ) -> dict:
        st = _call_build_state(build_state, x_all, pack=pack, time=time, required=required)
        if st is None:
            st = {}
        if not isinstance(st, dict):
            raise TypeError("build_state must return a dict.")
        if required is None or pack is None:
            return st

        if not hasattr(pack, "slices") or q_var not in pack.slices:
            return st

        x_all = np.asarray(x_all, dtype=float).reshape(-1)
        s, e = pack.slices[q_var]
        q_all = np.asarray(x_all[s:e], dtype=float).reshape(-1)
        q_dim_total = int(q_all.size)

        steps = 1
        if time is not None and hasattr(time, "N"):
            try:
                steps = int(time.N) + 1
            except Exception:
                steps = 1

        chunked = bool(steps > 1 and q_dim_total % steps == 0)
        nq = int(q_dim_total // steps) if chunked else int(q_dim_total)

        out = dict(st)

        for key in required:
            if not isinstance(key, StateKey):
                continue
            if key.owner != owner or key.dtype != dtype:
                continue

            k = int(key.k)

            if key.field == field:
                if chunked:
                    if k < 0 or k >= steps:
                        raise ValueError(f"Requested joint q at k={k}, but time steps are 0..{steps - 1}.")
                    start = k * nq
                    out.setdefault(key, q_all[start : start + nq].copy())
                else:
                    if k != 0:
                        raise ValueError(
                            f"Requested joint q at k={k}, but variable '{q_var}' is not time-chunked "
                            f"(dim={q_dim_total}, steps={steps})."
                        )
                    out.setdefault(key, q_all.copy())

            elif key.field == jac_name:
                if chunked:
                    if k < 0 or k >= steps:
                        raise ValueError(f"Requested joint Jacobian at k={k}, but time steps are 0..{steps - 1}.")
                    start = k * nq
                    J = np.zeros((nq, q_dim_total), dtype=float)
                    J[:, start : start + nq] = np.eye(nq, dtype=float)
                    out.setdefault(key, J)
                else:
                    if k != 0:
                        raise ValueError(
                            f"Requested joint Jacobian at k={k}, but variable '{q_var}' is not time-chunked "
                            f"(dim={q_dim_total}, steps={steps})."
                        )
                    out.setdefault(key, np.eye(q_dim_total, dtype=float))

        return out

    return wrapped
