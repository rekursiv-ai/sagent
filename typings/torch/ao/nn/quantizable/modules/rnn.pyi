from typing import Any, Self

from torch import Tensor

import torch

"""
We will recreate all the RNN modules as we require the modules to be decomposed
into its building blocks to be able to observe.
"""
__all__ = ["LSTM", "LSTMCell"]

class LSTMCell(torch.nn.Module):
    _FLOAT_MODULE = torch.nn.LSTMCell
    __constants__ = ...
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bias: bool = ...,
        device=...,
        dtype=...,
        *,
        split_gates=...,
    ) -> None: ...
    def forward(
        self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = ...
    ) -> tuple[Tensor, Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]: ...
    def initialize_hidden(
        self, batch_size: int, is_quantized: bool = ...
    ) -> tuple[Tensor, Tensor]: ...
    @classmethod
    def from_params(cls, wi, wh, bi=..., bh=..., split_gates=...) -> Self: ...
    @classmethod
    def from_float(
        cls, other, use_precomputed_fake_quant=..., split_gates=...
    ) -> Self: ...

class _LSTMSingleLayer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bias: bool = ...,
        device=...,
        dtype=...,
        *,
        split_gates=...,
    ) -> None: ...
    def forward(
        self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = ...
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]: ...
    @classmethod
    def from_params(cls, *args, **kwargs) -> Self: ...

class _LSTMLayer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bias: bool = ...,
        batch_first: bool = ...,
        bidirectional: bool = ...,
        device=...,
        dtype=...,
        *,
        split_gates=...,
    ) -> None: ...
    def forward(
        self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = ...
    ) -> tuple[Tensor | Any, tuple[Any | Tensor | None, Any | Tensor | None]]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Tensor | Any, tuple[Any | Tensor | None, Any | Tensor | None]]: ...
    @classmethod
    def from_float(cls, other, layer_idx=..., qconfig=..., **kwargs) -> Self: ...

class LSTM(torch.nn.Module):
    _FLOAT_MODULE = torch.nn.LSTM
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = ...,
        bias: bool = ...,
        batch_first: bool = ...,
        dropout: float = ...,
        bidirectional: bool = ...,
        device=...,
        dtype=...,
        *,
        split_gates: bool = ...,
    ) -> None: ...
    def forward(
        self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = ...
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]: ...
    @classmethod
    def from_float(cls, other, qconfig=..., split_gates=...): ...
    @classmethod
    def from_observed(cls, other): ...
