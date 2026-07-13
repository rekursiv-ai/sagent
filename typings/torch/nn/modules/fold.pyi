from torch import Tensor
from torch.nn.common_types import _size_any_t

from .module import Module

__all__ = ["Fold", "Unfold"]

class Fold(Module):
    ()  # noqa: B018 -- mirrors upstream module-level expression

    __constants__ = ...
    output_size: _size_any_t
    kernel_size: _size_any_t
    dilation: _size_any_t
    padding: _size_any_t
    stride: _size_any_t
    def __init__(
        self,
        output_size: _size_any_t,
        kernel_size: _size_any_t,
        dilation: _size_any_t = ...,
        padding: _size_any_t = ...,
        stride: _size_any_t = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Unfold(Module):
    ()  # noqa: B018 -- mirrors upstream module-level expression

    __constants__ = ...
    kernel_size: _size_any_t
    dilation: _size_any_t
    padding: _size_any_t
    stride: _size_any_t
    def __init__(
        self,
        kernel_size: _size_any_t,
        dilation: _size_any_t = ...,
        padding: _size_any_t = ...,
        stride: _size_any_t = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...
