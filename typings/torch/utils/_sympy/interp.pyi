from typing import Any

import functools

from sympy.logic.boolalg import Boolean as SympyBoolean

import sympy

"""
This is a simple interpreter for Sympy expressions that dispatches to
classes following the torch._inductor.virtualized calling convention.
For directness, the interpreter takes the handler directly rather than
consulting the TLS.  It does not use most of the methods on the full
handler; only those with corresponding Sympy expressions.  To see an example
of a full handler, see torch.utils._sympy.value_ranges.ValueRangeAnalysis.
"""
log = ...

@functools.cache
def handlers() -> dict[
    type[
        Or
        | And
        | Not
        | IntTrueDiv
        | FloatTrueDiv
        | FloorDiv
        | CleanDiv
        | TruncToFloat
        | Where
        | Add
        | Mul
        | FloatPow
        | PowByNatural
        | Pow
        | torch.utils._sympy.functions.Mod
        | PythonMod
        | sympy.core.mod.Mod
        | Abs
        | log
        | exp
        | sympy.functions.elementary.miscellaneous.Min
        | sympy.functions.elementary.miscellaneous.Max
        | torch.utils._sympy.functions.Min
        | torch.utils._sympy.functions.Max
        | ModularIndexing
        | Piecewise
        | Identity
        | IsNonOverlappingAndDenseIndicator
        | RoundDecimal
        | OpaqueUnaryFn
        | BitwiseFn
    ]
    | Eq
    | Ne
    | Lt
    | Gt
    | Le
    | Ge
    | Any,
    str,
]: ...

ASSOCIATIVE_OPS = ...
_nil = ...

def sympy_interp(
    analysis,
    env: dict[sympy.Symbol, Any],
    expr: sympy.Expr | SympyBoolean,
    *,
    index_dtype=...,
    missing_handler=...,
) -> Any: ...
