from collections.abc import Callable
from typing import Any

from sympy import Expr
from torch.utils._sympy.value_ranges import SymPyValueRangeAnalysis, ValueRanges

import sympy
import torch

from .loop_body import LoopBody, LoopBodyBlock
from .ops_handler import DefaultHandler, ReductionType, StoreMode
from .utils import cache_on_self

log = ...

class BoundVars:
    def __init__(self, loop_body: LoopBody) -> None: ...
    @cache_on_self
    def get_bounds(self) -> dict[torch.fx.Node, ValueRanges[Expr]]: ...
    def swap_submodules(
        self, submodules: dict[str, Callable[..., Any]]
    ) -> dict[str, Callable[..., ValueRanges[Expr]]]: ...
    def masked_subblock(
        self,
        subblock: LoopBodyBlock,
        env: dict[torch.fx.Node, ValueRanges[Expr]],
        mask: Any,
        value: Any,
        submodules: dict[str, Callable[..., Any]],
    ) -> ValueRanges[Expr]: ...
    def set_indirect(self, old: Expr, new: ValueRanges[Expr]) -> ValueRanges[Expr]: ...
    def get_index(self, name: str) -> ValueRanges[Expr]: ...

class ValueRangeAnalysis(SymPyValueRangeAnalysis, DefaultHandler):
    def __init__(self) -> None: ...
    @staticmethod
    def bool_handler(*args: Any, **kwargs: Any) -> ValueRanges[Any]: ...
    def load(self, name: str, index: sympy.Expr) -> ValueRanges[Any]: ...
    def store(
        self, name: str, index: sympy.Expr, value: Any, mode: StoreMode = ...
    ) -> None: ...
    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Any,
    ) -> ValueRanges[Any]: ...
    @classmethod
    def index_expr(cls, index: Any, dtype: torch.dtype) -> ValueRanges[Any]: ...
    @staticmethod
    def to_dtype(
        x: Any,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = ...,
        use_compute_types: bool = ...,
    ) -> ValueRanges[Any]: ...
    @staticmethod
    def square(x: Any) -> ValueRanges[Any]: ...
    @staticmethod
    def neg(x: Any) -> ValueRanges[Any]: ...
    @classmethod
    def truncdiv(cls, a: Any, b: Any) -> ValueRanges[Any]: ...
    @classmethod
    def sub(cls, a: Any, b: Any) -> ValueRanges[Any]: ...
