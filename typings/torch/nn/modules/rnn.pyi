from typing import Any, overload
from typing_extensions import deprecated

from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence

import torch

from .module import Module

__all__ = [
    "GRU",
    "LSTM",
    "RNN",
    "GRUCell",
    "LSTMCell",
    "RNNBase",
    "RNNCell",
    "RNNCellBase",
]
_rnn_impls = ...

@deprecated(
    "`apply_permutation` is deprecated, please use `tensor.index_select(dim, permutation)` instead",
    category=FutureWarning,
)
def apply_permutation(
    tensor: Tensor, permutation: Tensor, dim: int = ...
) -> Tensor: ...

class RNNBase(Module):
    __constants__ = ...
    __jit_unused_properties__ = ...
    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool
    proj_size: int
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
    ) -> None: ...
    def __setattr__(self, attr, value) -> None: ...
    def flatten_parameters(self) -> None: ...
    def reset_parameters(self) -> None: ...
    def check_input(self, input: Tensor, batch_sizes: Tensor | None) -> None: ...
    def get_expected_hidden_size(
        self, input: Tensor, batch_sizes: Tensor | None
    ) -> tuple[int, int, int]: ...
    def check_hidden_size(
        self, hx: Tensor, expected_hidden_size: tuple[int, int, int], msg: str = ...
    ) -> None: ...
    def check_forward_args(
        self, input: Tensor, hidden: Tensor, batch_sizes: Tensor | None
    ) -> None: ...
    def permute_hidden(self, hx: Tensor, permutation: Tensor | None) -> Tensor: ...
    def extra_repr(self) -> str: ...
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, d) -> None: ...
    @property
    def all_weights(self) -> list[list[Parameter]]: ...

class RNN(RNNBase):
    @overload
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = ...,
        nonlinearity: str = ...,
        bias: bool = ...,
        batch_first: bool = ...,
        dropout: float = ...,
        bidirectional: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    @overload
    def __init__(self, *args, **kwargs) -> None: ...
    def __init__(self, *args, **kwargs) -> None: ...
    @overload
    @torch._jit_internal._overload_method
    def forward(
        self, input: Tensor, hx: Tensor | None = ...
    ) -> tuple[Tensor, Tensor]: ...
    @overload
    @torch._jit_internal._overload_method
    def forward(
        self, input: PackedSequence, hx: Tensor | None = ...
    ) -> tuple[PackedSequence, Tensor]: ...
    def forward(
        self, input, hx=...
    ) -> tuple[PackedSequence, Tensor] | tuple[Any, Tensor]: ...

class LSTM(RNNBase):
    @overload
    def __init__(
        self,
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
    ) -> None: ...
    @overload
    def __init__(self, *args, **kwargs) -> None: ...
    def __init__(self, *args, **kwargs) -> None: ...
    def get_expected_cell_size(
        self, input: Tensor, batch_sizes: Tensor | None
    ) -> tuple[int, int, int]: ...
    def check_forward_args(
        self, input: Tensor, hidden: tuple[Tensor, Tensor], batch_sizes: Tensor | None
    ) -> None: ...
    def permute_hidden(
        self, hx: tuple[Tensor, Tensor], permutation: Tensor | None
    ) -> tuple[Tensor, Tensor]: ...
    @overload
    @torch._jit_internal._overload_method
    def forward(
        self, input: Tensor, hx: tuple[Tensor, Tensor] | None = ...
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]: ...
    @overload
    @torch._jit_internal._overload_method
    def forward(
        self, input: PackedSequence, hx: tuple[Tensor, Tensor] | None = ...
    ) -> tuple[PackedSequence, tuple[Tensor, Tensor]]: ...
    def forward(
        self, input, hx=...
    ) -> (
        tuple[PackedSequence, tuple[Tensor, Tensor]] | tuple[Any, tuple[Tensor, Tensor]]
    ): ...

class GRU(RNNBase):
    @overload
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
    ) -> None: ...
    @overload
    def __init__(self, *args, **kwargs) -> None: ...
    def __init__(self, *args, **kwargs) -> None: ...
    @overload
    @torch._jit_internal._overload_method
    def forward(
        self, input: Tensor, hx: Tensor | None = ...
    ) -> tuple[Tensor, Tensor]: ...
    @overload
    @torch._jit_internal._overload_method
    def forward(
        self, input: PackedSequence, hx: Tensor | None = ...
    ) -> tuple[PackedSequence, Tensor]: ...
    def forward(
        self, input, hx=...
    ) -> tuple[PackedSequence, Tensor] | tuple[Any, Tensor]: ...

class RNNCellBase(Module):
    __constants__ = ...
    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: Tensor
    weight_hh: Tensor
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        num_chunks: int,
        device=...,
        dtype=...,
    ) -> None: ...
    def extra_repr(self) -> str: ...
    def reset_parameters(self) -> None: ...

class RNNCell(RNNCellBase):
    __constants__ = ...
    nonlinearity: str
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = ...,
        nonlinearity: str = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor, hx: Tensor | None = ...) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class LSTMCell(RNNCellBase):
    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = ..., device=..., dtype=...
    ) -> None: ...
    def forward(
        self, input: Tensor, hx: tuple[Tensor, Tensor] | None = ...
    ) -> tuple[Tensor, Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]: ...

class GRUCell(RNNCellBase):
    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = ..., device=..., dtype=...
    ) -> None: ...
    def forward(self, input: Tensor, hx: Tensor | None = ...) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
