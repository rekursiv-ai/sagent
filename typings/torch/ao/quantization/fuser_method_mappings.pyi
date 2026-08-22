from collections.abc import Callable
from typing import Any

from torch import nn
from torch.ao.quantization.utils import Pattern

__all__ = [
    "fuse_conv_bn",
    "fuse_conv_bn_relu",
    "fuse_convtranspose_bn",
    "fuse_linear_bn",
    "get_fuser_method",
    "get_fuser_method_new",
]

def fuse_conv_bn(is_qat, conv, bn) -> ConvBn1d | ConvBn2d | ConvBn3d: ...
def fuse_conv_bn_relu(
    is_qat, conv, bn, relu
) -> (
    ConvBnReLU1d | ConvBnReLU2d | ConvBnReLU3d | ConvReLU1d | ConvReLU2d | ConvReLU3d
): ...
def fuse_linear_bn(is_qat, linear, bn) -> LinearBn1d: ...
def fuse_convtranspose_bn(is_qat, convt, bn): ...

_DEFAULT_OP_LIST_TO_FUSER_METHOD: dict[tuple, nn.Sequential | Callable] = ...

def get_fuser_method(op_list, additional_fuser_method_mapping=...): ...
def get_fuser_method_new(
    op_pattern: Pattern, fuser_method_mapping: dict[Pattern, nn.Sequential | Callable]
) -> Sequential | Callable[..., Any]: ...
