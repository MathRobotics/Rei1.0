from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import numpy as np

from .term import VariablePack, Cost, Expr

Array = np.ndarray


@dataclass
class Problem:
    """A collection of (expr, cost) terms assembled into one stacked LS problem."""

    variables: VariablePack
    terms: List[Tuple[Expr, Cost]]

    _last_rev: int = -1
    _last_time_rev: int = -1
    _last_req_sig: int = 0
    _last_r: Optional[Array] = None
    _last_J: Optional[Array] = None

    def _required_sig(self, required) -> int:
        if required is None:
            return 0
        return hash(frozenset(required))

    def linearize(self, *, ctx: Any = None, time: Any = None, required=None) -> Tuple[Array, Array]:
        rev = int(getattr(self.variables, "revision", 0))
        time_rev = int(getattr(time, "revision", 0)) if time is not None else 0
        req_sig = self._required_sig(required)

        if (
            self._last_r is not None
            and self._last_J is not None
            and rev == self._last_rev
            and time_rev == self._last_time_rev
            and req_sig == self._last_req_sig
        ):
            return self._last_r, self._last_J

        rs: List[Array] = []
        Js: List[Array] = []

        n_total = int(self.variables.n_total)
        slices = self.variables.slices

        if len(self.terms) == 0:
            raise ValueError("Problem.linearize: no terms.")

        for expr, cost in self.terms:
            r, blocks = expr.eval(ctx)  # (m,), [ (m, dim(var_i)), ... ]
            r2, blocks2 = cost.apply(r, blocks)

            r2 = np.asarray(r2, dtype=float).reshape(-1)
            m = int(r2.size)

            Jg = np.zeros((m, n_total), dtype=float)

            if len(blocks2) != len(expr.vars):
                raise ValueError(
                    f"Problem.linearize: len(blocks) mismatch in term '{expr.name}'. "
                    f"blocks={len(blocks2)}, vars={len(expr.vars)}"
                )

            for v, B in zip(expr.vars, blocks2):
                B = np.asarray(B, dtype=float)

                if B.ndim != 2 or B.shape[0] != m:
                    raise ValueError(
                        f"Problem.linearize: row mismatch in term '{expr.name}', var '{v.name}'. "
                        f"r has m={m}, block has {B.shape}."
                    )

                if v.name not in slices:
                    raise ValueError(
                        f"Problem.linearize: var '{v.name}' not found in VariablePack (term '{expr.name}')."
                    )

                s, e = slices[v.name]
                nv = e - s

                if B.shape[1] != nv:
                    raise ValueError(
                        f"Problem.linearize: col mismatch in term '{expr.name}', var '{v.name}'. "
                        f"expected block (m,{nv}), got {B.shape}."
                    )

                Jg[:, s:e] = B

            rs.append(r2)
            Js.append(Jg)

        r_all = np.concatenate(rs, axis=0) if rs else np.zeros((0,), dtype=float)
        J_all = np.vstack(Js) if Js else np.zeros((0, n_total), dtype=float)

        self._last_rev = rev
        self._last_time_rev = time_rev
        self._last_req_sig = req_sig
        self._last_r = r_all
        self._last_J = J_all

        return r_all, J_all

    def cost_value(self, *, ctx: Any = None, time: Any = None, required=None) -> float:
        r_all, _ = self.linearize(ctx=ctx, time=time, required=required)
        return float(r_all @ r_all)
