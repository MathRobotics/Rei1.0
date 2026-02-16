from __future__ import annotations

from .nodes import (
    ConstantExpr,
    GetStateExpr,
    GetVarExpr,
    HingeExpr,
    RepeatConstantExpr,
    StackExpr,
    SubExpr,
    TimeDiffExpr,
    TrajectoryVarDerivativesExpr,
    TrajectoryVarExpr,
)
from .registry import ExprRegister
from .types import (
    Array,
    DirectVectorExpr,
    Expr,
    RuntimeContext,
    Variable,
    VariablePack,
    pack,
    total_dim,
)


def register_stdlib(expr_register: ExprRegister) -> None:
    # Lazy import to avoid import cycle: core.expr -> stdlib -> dsl -> expr.
    from .stdlib import register_stdlib as _register_stdlib

    _register_stdlib(expr_register)

__all__ = [
    "Array",
    "ExprRegister",
    "register_stdlib",
    "Variable",
    "pack",
    "total_dim",
    "VariablePack",
    "RuntimeContext",
    "Expr",
    "DirectVectorExpr",
    "GetStateExpr",
    "GetVarExpr",
    "TrajectoryVarExpr",
    "TrajectoryVarDerivativesExpr",
    "TimeDiffExpr",
    "ConstantExpr",
    "RepeatConstantExpr",
    "SubExpr",
    "StackExpr",
    "HingeExpr",
]
