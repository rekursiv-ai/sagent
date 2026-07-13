from typing import Any

from torch import Tensor
from torch.nn.parameter import UninitializedParameter

from .lazy import LazyModuleMixin
from .module import Module

__all__ = ["Bilinear", "Identity", "LazyLinear", "Linear"]

class Identity(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor, *args: Any, **kwargs: Any) -> Tensor: ...

class Linear(Module):
    __constants__ = ...
    in_features: int
    out_features: int
    weight: Tensor
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class NonDynamicallyQuantizableLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...

class Bilinear(Module):
    __constants__ = ...
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor
    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input1: Tensor, input2: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class LazyLinear(LazyModuleMixin, Linear):
    cls_to_become = ...
    weight: UninitializedParameter
    bias: UninitializedParameter
    def __init__(
        self, out_features: int, bias: bool = ..., device=..., dtype=...
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def initialize_parameters(self, input) -> None: ...
