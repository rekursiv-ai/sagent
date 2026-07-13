from typing import (
    Optional as Optional,
    Union as Union,
)

from torch import (
    SymInt as SymInt,
    Tensor as Tensor,
)
from torch._C import (
    _add_docstr as _add_docstr,
    _nested as _nested,
)
from torch.types import (
    _device as Device,
    _dtype as DType,
)

from ._internal.nested_tensor import _rebuild_njt as _rebuild_njt

__all__ = [
    "as_nested_tensor",
    "masked_select",
    "narrow",
    "nested_tensor",
    "nested_tensor_from_jagged",
    "to_padded_tensor",
]

def as_nested_tensor(
    ts: Tensor | list[Tensor] | tuple[Tensor, ...],
    dtype: DType | None = ...,
    device: Device | None = ...,
    layout=...,
) -> Tensor: ...

to_padded_tensor = ...

def nested_tensor(
    tensor_list, *, dtype=..., layout=..., device=..., requires_grad=..., pin_memory=...
) -> Tensor: ...
def narrow(
    tensor: Tensor, dim: int, start: int | Tensor, length: int | Tensor, layout=...
) -> Tensor: ...
def nested_tensor_from_jagged(
    values: Tensor,
    offsets: Tensor | None = ...,
    lengths: Tensor | None = ...,
    jagged_dim: int | None = ...,
    min_seqlen: int | None = ...,
    max_seqlen: int | None = ...,
) -> Tensor: ...
def masked_select(tensor: Tensor, mask: Tensor) -> Tensor: ...
