from torch import Tensor

from .module import Module

__all__ = ["Embedding", "EmbeddingBag"]

class Embedding(Module):
    __constants__ = ...
    num_embeddings: int
    embedding_dim: int
    padding_idx: int | None
    max_norm: float | None
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    freeze: bool
    sparse: bool
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = ...,
        max_norm: float | None = ...,
        norm_type: float = ...,
        scale_grad_by_freq: bool = ...,
        sparse: bool = ...,
        _weight: Tensor | None = ...,
        _freeze: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...
    @classmethod
    def from_pretrained(
        cls,
        embeddings,
        freeze=...,
        padding_idx=...,
        max_norm=...,
        norm_type=...,
        scale_grad_by_freq=...,
        sparse=...,
    ) -> Self: ...

class EmbeddingBag(Module):
    __constants__ = ...
    num_embeddings: int
    embedding_dim: int
    max_norm: float | None
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    mode: str
    sparse: bool
    include_last_offset: bool
    padding_idx: int | None
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: float | None = ...,
        norm_type: float = ...,
        scale_grad_by_freq: bool = ...,
        mode: str = ...,
        sparse: bool = ...,
        _weight: Tensor | None = ...,
        include_last_offset: bool = ...,
        padding_idx: int | None = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(
        self,
        input: Tensor,
        offsets: Tensor | None = ...,
        per_sample_weights: Tensor | None = ...,
    ) -> Tensor: ...
    def extra_repr(self) -> str: ...
    @classmethod
    def from_pretrained(
        cls,
        embeddings: Tensor,
        freeze: bool = ...,
        max_norm: float | None = ...,
        norm_type: float = ...,
        scale_grad_by_freq: bool = ...,
        mode: str = ...,
        sparse: bool = ...,
        include_last_offset: bool = ...,
        padding_idx: int | None = ...,
    ) -> EmbeddingBag: ...
