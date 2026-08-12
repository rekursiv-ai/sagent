from typing import Any
from typing import Literal
from typing_extensions import deprecated

from torch import Tensor
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.parameter import UninitializedParameter

from .lazy import LazyModuleMixin
from .module import Module

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
]
convolution_notes = ...

class _ConvNd(Module):
    __constants__ = ...
    __annotations__ = ...
    in_channels: int
    _reversed_padding_repeated_twice: list[int]
    out_channels: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: str | tuple[int, ...]
    dilation: tuple[int, ...]
    transposed: bool
    output_padding: tuple[int, ...]
    groups: int
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"]
    weight: Tensor
    bias: Tensor | None
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: str | tuple[int, ...],
        dilation: tuple[int, ...],
        transposed: bool,
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"],
        device=...,
        dtype=...,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def extra_repr(self) -> str: ...
    def __setstate__(self, state) -> None: ...

class Conv1d(_ConvNd):
    __doc__ = ...
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = ...,
        padding: str | _size_1_t = ...,
        dilation: _size_1_t = ...,
        groups: int = ...,
        bias: bool = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class Conv2d(_ConvNd):
    __doc__ = ...
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = ...,
        padding: str | _size_2_t = ...,
        dilation: _size_2_t = ...,
        groups: int = ...,
        bias: bool = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    def _conv_forward(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
    ) -> Tensor: ...

class Conv3d(_ConvNd):
    __doc__ = ...
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = ...,
        padding: str | _size_3_t = ...,
        dilation: _size_3_t = ...,
        groups: int = ...,
        bias: bool = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class _ConvTransposeNd(_ConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        device=...,
        dtype=...,
    ) -> None: ...

class ConvTranspose1d(_ConvTransposeNd):
    __doc__ = ...
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = ...,
        padding: _size_1_t = ...,
        output_padding: _size_1_t = ...,
        groups: int = ...,
        bias: bool = ...,
        dilation: _size_1_t = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor, output_size: list[int] | None = ...) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class ConvTranspose2d(_ConvTransposeNd):
    __doc__ = ...
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = ...,
        padding: _size_2_t = ...,
        output_padding: _size_2_t = ...,
        groups: int = ...,
        bias: bool = ...,
        dilation: _size_2_t = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor, output_size: list[int] | None = ...) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class ConvTranspose3d(_ConvTransposeNd):
    __doc__ = ...
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = ...,
        padding: _size_3_t = ...,
        output_padding: _size_3_t = ...,
        groups: int = ...,
        bias: bool = ...,
        dilation: _size_3_t = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor, output_size: list[int] | None = ...) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class _ConvTransposeMixin(_ConvTransposeNd):
    @deprecated(
        "`_ConvTransposeMixin` is a deprecated internal class. Please consider using public APIs.",
        category=FutureWarning,
    )
    def __init__(self, *args, **kwargs) -> None: ...

class _LazyConvXdMixin(LazyModuleMixin):
    groups: int
    transposed: bool
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    weight: UninitializedParameter
    bias: UninitializedParameter
    def reset_parameters(self) -> None: ...
    def initialize_parameters(self, input: Tensor, *args, **kwargs) -> None: ...

class LazyConv1d(_LazyConvXdMixin, Conv1d):
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = ...,
        padding: _size_1_t = ...,
        dilation: _size_1_t = ...,
        groups: int = ...,
        bias: bool = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...

class LazyConv2d(_LazyConvXdMixin, Conv2d):
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = ...,
        padding: _size_2_t = ...,
        dilation: _size_2_t = ...,
        groups: int = ...,
        bias: bool = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...

class LazyConv3d(_LazyConvXdMixin, Conv3d):
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = ...,
        padding: _size_3_t = ...,
        dilation: _size_3_t = ...,
        groups: int = ...,
        bias: bool = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...

class LazyConvTranspose1d(_LazyConvXdMixin, ConvTranspose1d):
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = ...,
        padding: _size_1_t = ...,
        output_padding: _size_1_t = ...,
        groups: int = ...,
        bias: bool = ...,
        dilation: _size_1_t = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...

class LazyConvTranspose2d(_LazyConvXdMixin, ConvTranspose2d):
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = ...,
        padding: _size_2_t = ...,
        output_padding: _size_2_t = ...,
        groups: int = ...,
        bias: bool = ...,
        dilation: int = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...

class LazyConvTranspose3d(_LazyConvXdMixin, ConvTranspose3d):
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = ...,
        padding: _size_3_t = ...,
        output_padding: _size_3_t = ...,
        groups: int = ...,
        bias: bool = ...,
        dilation: _size_3_t = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...
