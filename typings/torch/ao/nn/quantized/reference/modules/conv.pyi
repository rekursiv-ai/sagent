from typing import Any, Literal

from torch import nn
from torch.nn.common_types import _size_1_t

import torch

from .utils import ReferenceQuantizedModule

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]

class _ConvNd(torch.nn.modules.conv._ConvNd, ReferenceQuantizedModule):
    __annotations__ = ...
    _IS_REFERENCE = ...
    @staticmethod
    def from_float(cls, float_conv, weight_qparams): ...

class Conv1d(_ConvNd, nn.Conv1d):
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
        weight_qparams: dict[str, Any] | None = ...,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...

class Conv2d(_ConvNd, nn.Conv2d):
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
        weight_qparams: dict[str, Any] | None = ...,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...

class Conv3d(_ConvNd, nn.Conv3d):
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
        weight_qparams: dict[str, Any] | None = ...,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...

class _ConvTransposeNd(_ConvNd, torch.nn.modules.conv._ConvTransposeNd):
    @staticmethod
    def from_float(cls, float_conv, weight_qparams): ...

class ConvTranspose1d(_ConvTransposeNd, nn.ConvTranspose1d):
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
        weight_qparams: dict[str, Any] | None = ...,
    ) -> None: ...
    def forward(
        self, x: torch.Tensor, output_size: list[int] | None = ...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...

class ConvTranspose2d(_ConvTransposeNd, nn.ConvTranspose2d):
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
        weight_qparams: dict[str, Any] | None = ...,
    ) -> None: ...
    def forward(
        self, x: torch.Tensor, output_size: list[int] | None = ...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...

class ConvTranspose3d(_ConvTransposeNd, nn.ConvTranspose3d):
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
        weight_qparams: dict[str, Any] | None = ...,
    ) -> None: ...
    def forward(
        self, x: torch.Tensor, output_size: list[int] | None = ...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...
