from torch import Tensor, nn

import torch
import torch.jit

__all__ = ["MultiheadAttention"]

class MultiheadAttention(nn.MultiheadAttention):
    _FLOAT_MODULE = nn.MultiheadAttention
    __constants__ = ...
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        bias: bool = ...,
        add_bias_kv: bool = ...,
        add_zero_attn: bool = ...,
        kdim: int | None = ...,
        vdim: int | None = ...,
        batch_first: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    @classmethod
    def from_float(cls, other): ...
    @torch.jit.unused
    def dequantize(self) -> _FLOAT_MODULE: ...
    @classmethod
    def from_observed(cls, other): ...
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = ...,
        need_weights: bool = ...,
        attn_mask: Tensor | None = ...,
        average_attn_weights: bool = ...,
        is_causal: bool = ...,
    ) -> tuple[Tensor, Tensor | None]: ...
