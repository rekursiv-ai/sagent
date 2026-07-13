from torch import Tensor, nn

__all__ = ["Embedding", "EmbeddingBag"]

class Embedding(nn.Embedding):
    _FLOAT_MODULE = nn.Embedding
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=...,
        max_norm=...,
        norm_type=...,
        scale_grad_by_freq=...,
        sparse=...,
        _weight=...,
        device=...,
        dtype=...,
        qconfig=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...
    def to_float(self) -> Embedding: ...

class EmbeddingBag(nn.EmbeddingBag):
    _FLOAT_MODULE = nn.EmbeddingBag
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        max_norm=...,
        norm_type=...,
        scale_grad_by_freq=...,
        mode=...,
        sparse=...,
        _weight=...,
        include_last_offset=...,
        padding_idx=...,
        qconfig=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input, offsets=..., per_sample_weights=...) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...
    def to_float(self) -> EmbeddingBag: ...
