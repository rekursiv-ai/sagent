from typing import Any, Self

from torch import Tensor, nn

from .utils import ReferenceQuantizedModule

__all__ = ["Embedding", "EmbeddingBag"]

class Embedding(nn.Embedding, ReferenceQuantizedModule):
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
        device=...,
        dtype=...,
        weight_qparams: dict[str, Any] | None = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, weight_qparams) -> Self: ...

class EmbeddingBag(nn.EmbeddingBag, ReferenceQuantizedModule):
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
        weight_qparams: dict[str, Any] | None = ...,
    ) -> None: ...
    def forward(
        self,
        input: Tensor,
        offsets: Tensor | None = ...,
        per_sample_weights: Tensor | None = ...,
    ) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @classmethod
    def from_float(
        cls, mod, weight_qparams, use_precomputed_fake_quant=...
    ) -> Self: ...
