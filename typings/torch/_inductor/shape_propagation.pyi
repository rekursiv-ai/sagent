from collections.abc import Callable, Sequence
from typing import Protocol

import functools

import sympy
import torch

from .virtualized import OpsValue

type BlockShapeType = Sequence[int | str] | None

class ShapeVar(Protocol):
    @property
    def shape(self) -> BlockShapeType: ...

type ShapeArg = ShapeVar | torch.types.Number | str | OpsValue | torch.dtype

@functools.lru_cache(None)
def get_broadcasted_shape(a: BlockShapeType, b: BlockShapeType) -> BlockShapeType: ...
def broadcast_shapes_for_args(args: Sequence[ShapeArg]) -> BlockShapeType: ...

class ShapePropagationOpsHandler:
    @staticmethod
    def constant(value: torch.types.Number, dtype: torch.dtype) -> BlockShapeType: ...
    @staticmethod
    def store_reduction(name: str, index: int, value: ShapeArg) -> None: ...
    @staticmethod
    def reduction(
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: str,
        value: ShapeArg | tuple[ShapeArg, ...],
    ) -> BlockShapeType | tuple[BlockShapeType, ...]: ...
    @staticmethod
    def store(
        name: str, index: int, value: ShapeArg, mode: str | None = ...
    ) -> None: ...
    @staticmethod
    def to_dtype(
        value: ShapeVar,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = ...,
        use_compute_types: bool = ...,
    ) -> BlockShapeType: ...
    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> BlockShapeType: ...
    @staticmethod
    def load_seed(name: str, offset: int) -> BlockShapeType: ...
    @staticmethod
    def indirect_indexing(
        var: ShapeArg, size: sympy.Expr | int, check: bool = ..., wrap_neg: bool = ...
    ) -> None: ...
    def __getattr__(self, name: str) -> Callable[..., BlockShapeType]: ...
    @staticmethod
    def device_assert_async(cond: ShapeArg, msg: str) -> None: ...
