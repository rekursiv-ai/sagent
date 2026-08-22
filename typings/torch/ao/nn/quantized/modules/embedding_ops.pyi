from typing import Any, Self

from torch import Tensor
from torch._jit_internal import Optional

import torch

__all__ = ["Embedding", "EmbeddingBag", "EmbeddingPackedParams"]

class EmbeddingPackedParams(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dtype=...) -> None: ...
    @torch.jit.export
    def set_weight(self, weight: torch.Tensor) -> None: ...
    def forward(self, x): ...
    def __repr__(self) -> Any: ...

class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = ...,
        max_norm: Optional[float] = ...,
        norm_type: float = ...,
        scale_grad_by_freq: bool = ...,
        sparse: bool = ...,
        _weight: Optional[Tensor] = ...,
        dtype=...,
    ) -> None: ...
    def forward(self, indices: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    def __repr__(self): ...
    def extra_repr(self) -> str: ...
    def set_weight(self, w: torch.Tensor) -> None: ...
    def weight(self) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Embedding: ...
    @classmethod
    def from_reference(cls, ref_embedding) -> Self: ...

class EmbeddingBag(Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: Optional[float] = ...,
        norm_type: float = ...,
        scale_grad_by_freq: bool = ...,
        mode: str = ...,
        sparse: bool = ...,
        _weight: Optional[Tensor] = ...,
        include_last_offset: bool = ...,
        dtype=...,
    ) -> None: ...
    def forward(
        self,
        indices: Tensor,
        offsets: Optional[Tensor] = ...,
        per_sample_weights: Optional[Tensor] = ...,
        compressed_indices_mapping: Optional[Tensor] = ...,
    ) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> EmbeddingBag: ...
    @classmethod
    def from_reference(cls, ref_embedding_bag) -> Self: ...
