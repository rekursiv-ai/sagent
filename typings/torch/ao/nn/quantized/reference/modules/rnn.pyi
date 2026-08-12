from typing import Self
from typing import Any

from torch import Tensor, nn

__all__ = [
    "GRU",
    "LSTM",
    "GRUCell",
    "LSTMCell",
    "RNNBase",
    "RNNCell",
    "RNNCellBase",
    "get_quantized_weight",
]

def get_quantized_weight(module, wn) -> Tensor | None: ...

class RNNCellBase(nn.RNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        num_chunks: int,
        device=...,
        dtype=...,
        weight_qparams_dict=...,
    ) -> None: ...
    def get_quantized_weight_ih(self) -> Tensor | None: ...
    def get_quantized_weight_hh(self) -> Tensor | None: ...
    def get_weight_ih(self) -> Tensor | None: ...
    def get_weight_hh(self) -> Tensor | None: ...

class RNNCell(RNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = ...,
        nonlinearity: str = ...,
        device=...,
        dtype=...,
        weight_qparams_dict: dict[str, Any] | None = ...,
    ) -> None: ...
    def forward(self, input: Tensor, hx: Tensor | None = ...) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, weight_qparams_dict) -> Self: ...

class LSTMCell(RNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = ...,
        device=...,
        dtype=...,
        weight_qparams_dict: dict[str, Any] | None = ...,
    ) -> None: ...
    def forward(
        self, input: Tensor, hx: tuple[Tensor, Tensor] | None = ...
    ) -> tuple[Tensor, Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]: ...
    @classmethod
    def from_float(
        cls, mod, weight_qparams_dict, use_precomputed_fake_quant=...
    ) -> Self: ...

class GRUCell(RNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = ...,
        device=...,
        dtype=...,
        weight_qparams_dict: dict[str, Any] | None = ...,
    ) -> None: ...
    def forward(self, input: Tensor, hx: Tensor | None = ...) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, weight_qparams_dict) -> Self: ...

class RNNBase(nn.RNNBase):
    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = ...,
        bias: bool = ...,
        batch_first: bool = ...,
        dropout: float = ...,
        bidirectional: bool = ...,
        proj_size: int = ...,
        device=...,
        dtype=...,
        weight_qparams_dict: dict[str, Any] | None = ...,
    ) -> None: ...

class LSTM(RNNBase):
    def __init__(self, *args, **kwargs) -> None: ...
    def permute_hidden(
        self, hx: tuple[Tensor, Tensor], permutation: Tensor | None
    ) -> tuple[Tensor, Tensor]: ...
    def get_expected_cell_size(
        self, input: Tensor, batch_sizes: Tensor | None
    ) -> tuple[int, int, int]: ...
    def check_forward_args(
        self, input: Tensor, hidden: tuple[Tensor, Tensor], batch_sizes: Tensor | None
    ) -> None: ...
    def get_quantized_weight_bias_dict(self) -> dict[Any, Any]: ...
    def get_flat_weights(self) -> list[Any]: ...
    def forward(
        self, input, hx=...
    ) -> (
        tuple[PackedSequence, tuple[Tensor, Tensor]]
        | tuple[Tensor, tuple[Tensor, Tensor]]
    ): ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> (
        tuple[PackedSequence, tuple[Tensor, Tensor]]
        | tuple[Tensor, tuple[Tensor, Tensor]]
    ): ...
    @classmethod
    def from_float(cls, mod, weight_qparams_dict) -> Self: ...

class GRU(RNNBase):
    def __init__(self, *args, **kwargs) -> None: ...
    def get_quantized_weight_bias_dict(self) -> dict[Any, Any]: ...
    def get_flat_weights(self) -> list[Any]: ...
    def forward(
        self, input, hx=...
    ) -> tuple[PackedSequence, Tensor] | tuple[Tensor, Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[PackedSequence, Tensor] | tuple[Tensor, Tensor]: ...
    @classmethod
    def from_float(cls, mod, weight_qparams_dict) -> Self: ...
