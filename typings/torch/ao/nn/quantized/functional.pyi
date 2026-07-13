from torch import Tensor
from torch.jit.annotations import BroadcastingList2

r"""Functional interface (quantized)."""
__all__ = [
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "avg_pool2d",
    "avg_pool3d",
    "celu",
    "clamp",
    "conv1d",
    "conv2d",
    "conv3d",
    "elu",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "interpolate",
    "leaky_relu",
    "linear",
    "max_pool1d",
    "max_pool2d",
    "threshold",
    "upsample",
    "upsample_bilinear",
    "upsample_nearest",
]

def avg_pool2d(
    input,
    kernel_size,
    stride=...,
    padding=...,
    ceil_mode=...,
    count_include_pad=...,
    divisor_override=...,
) -> Tensor: ...
def avg_pool3d(
    input,
    kernel_size,
    stride=...,
    padding=...,
    ceil_mode=...,
    count_include_pad=...,
    divisor_override=...,
) -> Tensor: ...
def adaptive_avg_pool2d(
    input: Tensor, output_size: BroadcastingList2[int]
) -> Tensor: ...
def adaptive_avg_pool3d(
    input: Tensor, output_size: BroadcastingList2[int]
) -> Tensor: ...
def conv1d(
    input,
    weight,
    bias,
    stride=...,
    padding=...,
    dilation=...,
    groups=...,
    padding_mode=...,
    scale=...,
    zero_point=...,
    dtype=...,
) -> Any: ...
def conv2d(
    input,
    weight,
    bias,
    stride=...,
    padding=...,
    dilation=...,
    groups=...,
    padding_mode=...,
    scale=...,
    zero_point=...,
    dtype=...,
) -> Any: ...
def conv3d(
    input,
    weight,
    bias,
    stride=...,
    padding=...,
    dilation=...,
    groups=...,
    padding_mode=...,
    scale=...,
    zero_point=...,
    dtype=...,
) -> Any: ...
def interpolate(
    input, size=..., scale_factor=..., mode=..., align_corners=...
) -> Tensor: ...
def linear(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = ...,
    scale: float | None = ...,
    zero_point: int | None = ...,
) -> Tensor: ...
def max_pool1d(
    input,
    kernel_size,
    stride=...,
    padding=...,
    dilation=...,
    ceil_mode=...,
    return_indices=...,
) -> Tensor: ...
def max_pool2d(
    input,
    kernel_size,
    stride=...,
    padding=...,
    dilation=...,
    ceil_mode=...,
    return_indices=...,
) -> Tensor: ...
def celu(
    input: Tensor, scale: float, zero_point: int, alpha: float = ...
) -> Tensor: ...
def leaky_relu(
    input: Tensor,
    negative_slope: float = ...,
    inplace: bool = ...,
    scale: float | None = ...,
    zero_point: int | None = ...,
) -> Tensor: ...
def hardtanh(
    input: Tensor, min_val: float = ..., max_val: float = ..., inplace: bool = ...
) -> Tensor: ...
def hardswish(input: Tensor, scale: float, zero_point: int) -> Tensor: ...
def threshold(input: Tensor, threshold: float, value: float) -> Tensor: ...
def elu(input: Tensor, scale: float, zero_point: int, alpha: float = ...) -> Tensor: ...
def hardsigmoid(input: Tensor, inplace: bool = ...) -> Tensor: ...
def clamp(input: Tensor, min_: float, max_: float) -> Tensor: ...
def upsample(
    input, size=..., scale_factor=..., mode=..., align_corners=...
) -> Tensor: ...
def upsample_bilinear(input, size=..., scale_factor=...) -> Tensor: ...
def upsample_nearest(input, size=..., scale_factor=...) -> Tensor: ...
