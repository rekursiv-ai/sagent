from typing import ClassVar, Literal

from torch import nn
from torch.nn.common_types import _size_1_t

import torch
import torch.ao.nn.intrinsic as nni

from .utils import WeightedQuantizedModule

r"""Quantized convolution modules."""
__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]
_SUPPORTED_PADDING = ...

class _ConvNd(WeightedQuantizedModule):
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
    def set_weight_bias(self, qweight, bias_float): ...
    def bias(self): ...
    def extra_repr(self) -> str: ...
    @torch.jit.export
    def __getstate__(
        self,
    ) -> tuple[
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        str | Any,
        Any,
        Any,
        float | Any,
        int | Any,
        bool,
    ]: ...
    @torch.jit.export
    def __setstate__(self, state) -> None: ...
    def __deepcopy__(self, memo) -> Self: ...
    def __copy__(self) -> Self: ...
    @classmethod
    def get_qconv(
        cls, mod, activation_post_process, weight_post_process=...
    ) -> Self: ...
    @staticmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point) -> Self: ...

class Conv1d(_ConvNd):
    _FLOAT_MODULE: ClassVar[type[nn.Conv1d]] = ...
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    _NNI_CONV_ADD_MODULE: ClassVar[type[nn.Module] | None] = ...
    _NNI_CONV_ADD_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
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
    ) -> None: ...
    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None: ...
    def weight(self) -> Any: ...
    def bias(self) -> Any: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...

class Conv2d(_ConvNd):
    _FLOAT_MODULE: ClassVar[type[nn.Conv2d]] = ...
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    _NNI_CONV_ADD_MODULE: ClassVar[type[nni.ConvAdd2d]] = ...
    _NNI_CONV_ADD_RELU_MODULE: ClassVar[type[nni.ConvAddReLU2d]] = ...
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
    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None: ...
    def weight(self) -> Any: ...
    def bias(self) -> Any: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...

class Conv3d(_ConvNd):
    _FLOAT_MODULE: ClassVar[type[nn.Conv3d]] = ...
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    _NNI_CONV_ADD_MODULE: ClassVar[type[nn.Module] | None] = ...
    _NNI_CONV_ADD_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
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
    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None: ...
    def weight(self) -> Any: ...
    def bias(self) -> Any: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...

class _ConvTransposeNd(_ConvNd):
    _FLOAT_MODULE: ClassVar[type[nn.modules.conv._ConvNd]]
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
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...
    @staticmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point): ...

class ConvTranspose1d(_ConvTransposeNd):
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
    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None: ...
    def weight(self) -> Any: ...
    def bias(self) -> Any: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point): ...

class ConvTranspose2d(_ConvTransposeNd):
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
    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None: ...
    def weight(self) -> Any: ...
    def bias(self) -> Any: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point): ...

class ConvTranspose3d(_ConvTransposeNd):
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
    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None: ...
    def weight(self) -> Any: ...
    def bias(self) -> Any: ...
    def forward(self, input) -> Any: ...
    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point): ...
