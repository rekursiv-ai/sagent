from typing import Any
import sympy
import torch

from .ir import ShapeAsConstantBuffer, TensorBox

def dense_idx_to_jagged_idx(
    batch_idx, seq_idx, offsets_loader, jagged_len
) -> tuple[Any, Any]: ...
def get_inverse_offsets(
    offsets: TensorBox, jagged_len: int | sympy.Expr, realize: bool = ...
) -> TensorBox | ShapeAsConstantBuffer: ...
def jagged_idx_to_dense_idx(
    jagged_idx,
    inverse_offsets_loader,
    offsets_loader,
    batch_size: int | sympy.Expr,
    max_seq_len: int | sympy.Expr,
    offsets_dtype: torch.dtype,
) -> tuple[sympy.Expr, sympy.Expr]: ...
def register_jagged_ops() -> None: ...
