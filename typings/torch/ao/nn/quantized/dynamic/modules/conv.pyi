from typing import Any
from typing import ClassVar, Literal

from torch import Tensor, nn
from torch.nn.common_types import _size_1_t

import torch.ao.nn.quantized as nnq

r"""Dynamically quantized convolution modules."""
__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]

class Conv1d(nnq.Conv1d):
    _FLOAT_MODULE: ClassVar[type[nn.Conv1d]] = ...
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    def __init__(
        self,
        in_channels: int,
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
        reduce_range=...,
    ) -> None: ...
    def forward(self, input: Tensor, reduce_range: bool = ...) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class Conv2d(nnq.Conv2d):
    _FLOAT_MODULE: ClassVar[type[nn.Conv2d]] = ...
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor, reduce_range: bool = ...) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class Conv3d(nnq.Conv3d):
    _FLOAT_MODULE: ClassVar[type[nn.Conv3d]] = ...
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor, reduce_range: bool = ...) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class ConvTranspose1d(nnq.ConvTranspose1d):
    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose1d]] = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        output_padding=...,
        groups=...,
        bias=...,
        dilation=...,
        padding_mode=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor, reduce_range: bool = ...) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class ConvTranspose2d(nnq.ConvTranspose2d):
    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose2d]] = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        output_padding=...,
        groups=...,
        bias=...,
        dilation=...,
        padding_mode=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor, reduce_range: bool = ...) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class ConvTranspose3d(nnq.ConvTranspose3d):
    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose3d]] = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        output_padding=...,
        groups=...,
        bias=...,
        dilation=...,
        padding_mode=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor, reduce_range: bool = ...) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
