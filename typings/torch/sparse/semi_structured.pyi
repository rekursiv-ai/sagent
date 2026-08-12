from torch import Tensor
from collections.abc import Callable
from typing import Any

import torch

__all__ = [
    "SparseSemiStructuredTensor",
    "SparseSemiStructuredTensorCUSPARSELT",
    "SparseSemiStructuredTensorCUTLASS",
    "to_sparse_semi_structured",
]
_SEMI_STRUCTURED_SPARSE_CONFIG = ...

class SparseSemiStructuredTensor(torch.Tensor):
    _DEFAULT_ALG_ID: int = ...
    _DTYPE_SHAPE_CONSTRAINTS: dict[torch.dtype, _SEMI_STRUCTURED_SPARSE_CONFIG]
    _FORCE_CUTLASS: bool = ...
    _FUSE_TRANSPOSE: bool = ...
    _PROTOTYPE_WARNING_SHOWN: bool = ...
    BACKEND: str
    SPARSE_DISPATCH: dict[Callable, Callable]
    packed: torch.Tensor | None
    meta: torch.Tensor | None
    packed_t: torch.Tensor | None
    meta_t: torch.Tensor | None
    compressed_swizzled_bitmask: torch.Tensor | None
    fuse_transpose_cusparselt: bool
    alg_id_cusparselt: int
    __slots__ = ...
    @staticmethod
    def __new__(
        cls,
        shape: torch.Size,
        packed: torch.Tensor | None,
        meta: torch.Tensor | None,
        packed_t: torch.Tensor | None,
        meta_t: torch.Tensor | None,
        compressed_swizzled_bitmask: torch.Tensor | None,
        fuse_transpose_cusparselt: bool = ...,
        alg_id_cusparselt: int = ...,
        requires_grad: bool = ...,
    ): ...
    def __tensor_flatten__(
        self,
    ) -> tuple[list[str], tuple[torch.Size, bool, int, bool]]: ...
    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors,
        tensor_meta: tuple[torch.Size, bool, int, bool],
        outer_size,
        outer_stride,
    ) -> torch.Tensor: ...
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> Any: ...
    def to_dense(self) -> Tensor: ...
    @classmethod
    def from_dense(
        cls, original_tensor: torch.Tensor
    ) -> SparseSemiStructuredTensor: ...

def to_sparse_semi_structured(
    original_tensor: torch.Tensor, transposed: bool = ...
) -> SparseSemiStructuredTensor: ...

class SparseSemiStructuredTensorCUTLASS(SparseSemiStructuredTensor):
    @classmethod
    def from_dense(
        cls, original_tensor: torch.Tensor
    ) -> SparseSemiStructuredTensorCUTLASS: ...
    def to_dense(self) -> Tensor: ...
    @classmethod
    def prune_dense_static_sort(
        cls, original_tensor: torch.Tensor, algorithm=...
    ) -> SparseSemiStructuredTensor: ...

class SparseSemiStructuredTensorCUSPARSELT(SparseSemiStructuredTensor):
    @classmethod
    def from_dense(
        cls, original_tensor: torch.Tensor
    ) -> SparseSemiStructuredTensorCUSPARSELT: ...
    @classmethod
    def prune_dense_static_sort(
        cls, original_tensor: torch.Tensor, algorithm=...
    ) -> SparseSemiStructuredTensor: ...
