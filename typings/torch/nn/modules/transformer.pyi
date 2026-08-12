from collections.abc import Callable
from typing import Any

from torch import Tensor

import torch

from .module import Module

__all__ = [
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
]

class Transformer(Module):
    def __init__(
        self,
        d_model: int = ...,
        nhead: int = ...,
        num_encoder_layers: int = ...,
        num_decoder_layers: int = ...,
        dim_feedforward: int = ...,
        dropout: float = ...,
        activation: str | Callable[[Tensor], Tensor] = ...,
        custom_encoder: Any | None = ...,
        custom_decoder: Any | None = ...,
        layer_norm_eps: float = ...,
        batch_first: bool = ...,
        norm_first: bool = ...,
        bias: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = ...,
        tgt_mask: Tensor | None = ...,
        memory_mask: Tensor | None = ...,
        src_key_padding_mask: Tensor | None = ...,
        tgt_key_padding_mask: Tensor | None = ...,
        memory_key_padding_mask: Tensor | None = ...,
        src_is_causal: bool | None = ...,
        tgt_is_causal: bool | None = ...,
        memory_is_causal: bool = ...,
    ) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @staticmethod
    def generate_square_subsequent_mask(
        sz: int, device: torch.device | None = ..., dtype: torch.dtype | None = ...
    ) -> Tensor: ...

class TransformerEncoder(Module):
    __constants__ = ...
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Module | None = ...,
        enable_nested_tensor: bool = ...,
        mask_check: bool = ...,
    ) -> None: ...
    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = ...,
        src_key_padding_mask: Tensor | None = ...,
        is_causal: bool | None = ...,
    ) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class TransformerDecoder(Module):
    __constants__ = ...
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Module | None = ...,
    ) -> None: ...
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = ...,
        memory_mask: Tensor | None = ...,
        tgt_key_padding_mask: Tensor | None = ...,
        memory_key_padding_mask: Tensor | None = ...,
        tgt_is_causal: bool | None = ...,
        memory_is_causal: bool = ...,
    ) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class TransformerEncoderLayer(Module):
    __constants__ = ...
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = ...,
        dropout: float = ...,
        activation: str | Callable[[Tensor], Tensor] = ...,
        layer_norm_eps: float = ...,
        batch_first: bool = ...,
        norm_first: bool = ...,
        bias: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def __setstate__(self, state) -> None: ...
    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = ...,
        src_key_padding_mask: Tensor | None = ...,
        is_causal: bool = ...,
    ) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class TransformerDecoderLayer(Module):
    __constants__ = ...
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = ...,
        dropout: float = ...,
        activation: str | Callable[[Tensor], Tensor] = ...,
        layer_norm_eps: float = ...,
        batch_first: bool = ...,
        norm_first: bool = ...,
        bias: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def __setstate__(self, state) -> None: ...
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = ...,
        memory_mask: Tensor | None = ...,
        tgt_key_padding_mask: Tensor | None = ...,
        memory_key_padding_mask: Tensor | None = ...,
        tgt_is_causal: bool = ...,
        memory_is_causal: bool = ...,
    ) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
