from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, overload

import sympy
import torch

from .ops_handler import DefaultHandler

"""This file implements the IndexPropagation ops handler, which wraps an
underlying handler to add a limited form of constant propagation, as well as
propagation of sympy expressions downstream of ops.index_expr calls.

For example, say we have the IR:

    tmp0 = ops.index_expr(x, torch.int32)
    tmp1 = ops.constant(2, torch.int32)
    tmp2 = ops.mul(tmp0, tmp1)
    tmp3 = ops.indirect_indexing(tmp2, x_size)
    tmp4 = ops.load("buf0", tmp3)

The underlying handler would just see:

    ops.load("buf0", x * 2)

This is limited by the set of operators handled in the sympy expression
printers. So simple operations like minimum and maximum cannot be translated to
SymPy expressions yet, despite sympy.Min and sympy.Max existing.

"""
type _ExprType = sympy.Expr | float | int | bool

def upper_bound(val: _ExprType) -> float | int | bool: ...

@dataclass
class TypedExpr:
    expr: _ExprType
    dtype: torch.dtype
    def is_constant(self) -> bool: ...
    def __post_init__(self) -> None: ...

class SymPyOps:
    @staticmethod
    def identity(value: Any) -> Any: ...
    @staticmethod
    def constant(value: float | bool, dtype: torch.dtype) -> TypedExpr: ...
    @staticmethod
    def index_expr(value: sympy.Expr | int, dtype: torch.dtype) -> TypedExpr: ...
    @staticmethod
    def to_dtype(
        value: TypedExpr,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = ...,
        use_compute_types: bool = ...,
    ) -> TypedExpr: ...
    @staticmethod
    def abs(x: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def square(x: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def add(x: TypedExpr, y: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def sub(x: TypedExpr, y: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def mul(x: TypedExpr, y: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def neg(x: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def floordiv(x: TypedExpr, y: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def mod(x: TypedExpr, y: TypedExpr) -> TypedExpr | None: ...
    @staticmethod
    def remainder(x: TypedExpr, y: TypedExpr) -> TypedExpr | None: ...
    @staticmethod
    def minimum(x: TypedExpr, y: TypedExpr) -> TypedExpr: ...
    @staticmethod
    def maximum(x: TypedExpr, y: TypedExpr) -> TypedExpr: ...

@dataclass
class IndexPropVar:
    value: Any
    is_symbolic: bool = ...
    @staticmethod
    def new_symbolic(expr: TypedExpr) -> IndexPropVar: ...
    def __post_init__(self) -> None: ...

type IndexPropResult = IndexPropVar | tuple[IndexPropResult, ...]

class IndexPropagation(DefaultHandler):
    def __init__(
        self,
        inner: Any,
        iter_ranges: dict[sympy.Symbol, sympy.Expr],
        indirect_var_ranges: dict[sympy.Symbol, sympy.Expr],
    ) -> None: ...
    def materialize_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> Any: ...
    def unwrap(self, a: Any | IndexPropVar) -> Any: ...
    def wrap(self, a) -> IndexPropResult: ...
    @overload
    def fallback(
        self,
        name: Literal["indirect_indexing"],
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> IndexPropVar: ...
    @overload
    def fallback(
        self, name: str, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> IndexPropResult: ...
    def fallback(
        self, name: str, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> IndexPropResult: ...
    def propagate_sympy(
        self, name: str, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> IndexPropResult: ...
    def statically_true(self, e) -> bool: ...
    def indirect_indexing(
        self, index: Any | IndexPropVar, size: Any, check: bool = ..., wrap_neg=...
    ) -> Any: ...
