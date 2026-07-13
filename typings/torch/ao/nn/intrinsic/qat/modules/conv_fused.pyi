from typing import ClassVar

from torch import nn

import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat

__all__ = [
    "ConvBn1d",
    "ConvBn2d",
    "ConvBn3d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBnReLU3d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "freeze_bn_stats",
    "update_bn_stats",
]
_BN_CLASS_MAP = ...

class _ConvBnNd(nn.modules.conv._ConvNd, nni._FusedModule):
    _version = ...
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
        eps=...,
        momentum=...,
        freeze_bn=...,
        qconfig=...,
        dim=...,
    ) -> None: ...
    def reset_running_stats(self) -> None: ...
    def reset_bn_parameters(self) -> None: ...
    def reset_parameters(self) -> None: ...
    def update_bn_stats(self) -> Self: ...
    def freeze_bn_stats(self) -> Self: ...
    def extra_repr(self) -> str: ...
    def forward(self, input) -> Tensor | Any: ...
    def train(self, mode=...) -> Self: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...
    def to_float(self): ...

class ConvBn1d(_ConvBnNd, nn.Conv1d):
    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm1d]] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FLOAT_MODULE: ClassVar[type[nn.Module]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]] = ...
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
        eps=...,
        momentum=...,
        freeze_bn=...,
        qconfig=...,
    ) -> None: ...

class ConvBnReLU1d(ConvBn1d):
    _FLOAT_MODULE: ClassVar[type[nn.Module]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm1d]] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FUSED_FLOAT_MODULE: ClassVar[type[nn.Module] | None] = ...
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
        eps=...,
        momentum=...,
        freeze_bn=...,
        qconfig=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...

class ConvReLU1d(nnqat.Conv1d, nni._FusedModule):
    _FLOAT_MODULE: ClassVar[type[nni.ConvReLU1d]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
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
        qconfig=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...

class ConvBn2d(_ConvBnNd, nn.Conv2d):
    _FLOAT_MODULE: ClassVar[type[nni.ConvBn2d]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
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
        eps=...,
        momentum=...,
        freeze_bn=...,
        qconfig=...,
    ) -> None: ...

class ConvBnReLU2d(ConvBn2d):
    _FLOAT_MODULE: ClassVar[type[nni.ConvBnReLU2d]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm2d]] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FUSED_FLOAT_MODULE: ClassVar[type[nni.ConvReLU2d] | None] = ...
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
        eps=...,
        momentum=...,
        freeze_bn=...,
        qconfig=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...

class ConvReLU2d(nnqat.Conv2d, nni._FusedModule):
    _FLOAT_MODULE: ClassVar[type[nn.Module]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
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
        qconfig=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...

class ConvBn3d(_ConvBnNd, nn.Conv3d):
    _FLOAT_MODULE: ClassVar[type[nni.ConvBn3d]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
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
        eps=...,
        momentum=...,
        freeze_bn=...,
        qconfig=...,
    ) -> None: ...

class ConvBnReLU3d(ConvBn3d):
    _FLOAT_MODULE: ClassVar[type[nni.ConvBnReLU3d]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm3d]] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.ReLU] | None] = ...
    _FUSED_FLOAT_MODULE: ClassVar[type[nni.ConvReLU3d] | None] = ...
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
        eps=...,
        momentum=...,
        freeze_bn=...,
        qconfig=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...

class ConvReLU3d(nnqat.Conv3d, nni._FusedModule):
    _FLOAT_MODULE: ClassVar[type[nni.ConvReLU3d]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
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
        qconfig=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...

def update_bn_stats(mod) -> None: ...
def freeze_bn_stats(mod) -> None: ...
