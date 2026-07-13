from typing import ClassVar, Literal

from torch import nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

__all__ = ["Conv1d", "Conv2d", "Conv3d"]

class _ConvNd(nn.modules.conv._ConvNd):
    _FLOAT_MODULE: ClassVar[type[nn.modules.conv._ConvNd]]
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
        qconfig=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @staticmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...
    def to_float(self) -> _ConvNd: ...

class Conv1d(_ConvNd, nn.Conv1d):
    _FLOAT_MODULE: ClassVar[type[nn.Conv1d]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]] = ...
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
        qconfig=...,
        device=...,
        dtype=...,
    ) -> None: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...

class Conv2d(_ConvNd, nn.Conv2d):
    _FLOAT_MODULE: ClassVar[type[nn.Conv2d]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]] = ...
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
        qconfig=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...

class Conv3d(_ConvNd, nn.Conv3d):
    _FLOAT_MODULE: ClassVar[type[nn.Conv3d]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]] = ...
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
        qconfig=...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...
