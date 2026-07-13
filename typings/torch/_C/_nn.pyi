from collections.abc import Sequence
from typing import Literal

from torch import Tensor
from torch.types import _int, _size

def adaptive_avg_pool2d(input: Tensor, output_size: _int | _size) -> Tensor: ...
def adaptive_avg_pool3d(input: Tensor, output_size: _int | _size) -> Tensor: ...
def adaptive_max_pool2d(
    input: Tensor, output_size: _int | _size
) -> tuple[Tensor, Tensor]: ...
def adaptive_max_pool3d(
    input: Tensor, output_size: _int | _size
) -> tuple[Tensor, Tensor]: ...
def avg_pool2d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
    ceil_mode: bool = ...,
    count_include_pad: bool = ...,
    divisor_override: int | None = ...,
) -> Tensor: ...
def avg_pool3d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
    ceil_mode: bool = ...,
    count_include_pad: bool = ...,
    divisor_override: int | None = ...,
) -> Tensor: ...
def binary_cross_entropy(
    input: Tensor, target: Tensor, weight: Tensor | None = ..., reduction: str = ...
) -> Tensor: ...
def col2im(
    input: Tensor,
    output_size: _int | _size,
    kernel_size: _int | _size,
    dilation: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
) -> Tensor: ...
def cross_entropy_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = ...,
    reduction: str = ...,
    ignore_index: int = ...,
    label_smoothing: float = ...,
) -> Tensor: ...
def elu(
    input: Tensor, alpha: float = ..., scale: float = ..., input_scale: float = ...
) -> Tensor: ...
def elu_(input: Tensor, alpha: float = ...) -> Tensor: ...
def fractional_max_pool2d(
    input: Tensor,
    kernel_size: _int | _size,
    output_size: _int | _size,
    _random_samples: Tensor,
) -> tuple[Tensor, Tensor]: ...
def fractional_max_pool3d(
    input: Tensor,
    kernel_size: _int | _size,
    output_size: _int | _size,
    _random_samples: Tensor,
) -> tuple[Tensor, Tensor]: ...
def gelu(input: Tensor, approximate: str = ...) -> Tensor: ...
def glu(input: Tensor, dim: int = ...) -> Tensor: ...
def hardsigmoid(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def hardsigmoid_(input: Tensor) -> Tensor: ...
def hardswish(input: Tensor) -> Tensor: ...
def hardswish_(input: Tensor) -> Tensor: ...
def hardtanh(
    input: Tensor,
    min_val: float = ...,
    max_val: float = ...,
    *,
    out: Tensor | None = ...,
) -> Tensor: ...
def hardtanh_(input: Tensor, min_val: float = ..., max_val: float = ...) -> Tensor: ...
def huber_loss(
    input: Tensor, target: Tensor, reduction: str = ..., delta: float = ...
) -> Tensor: ...
def leaky_relu(
    input: Tensor, negative_slope: float = ..., *, out: Tensor | None = ...
) -> Tensor: ...
def leaky_relu_(input: Tensor, negative_slope: float = ...) -> Tensor: ...
def linear(input: Tensor, weight: Tensor, bias: Tensor | None = ...) -> Tensor: ...
def log_sigmoid(input: Tensor) -> Tensor: ...
def max_pool2d_with_indices(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: bool = ...,
) -> tuple[Tensor, Tensor]: ...
def max_pool3d_with_indices(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: bool = ...,
) -> tuple[Tensor, Tensor]: ...
def max_unpool2d(
    input: Tensor, indices: Tensor, output_size: Sequence[int] | None
) -> Tensor: ...
def max_unpool3d(
    input: Tensor,
    indices: Tensor,
    output_size: Sequence[int] | None,
    stride: _int | _size,
    padding: _int | _size,
) -> Tensor: ...
def one_hot(tensor: Tensor, num_classes: int = ...) -> Tensor: ...
def pad(
    input: Tensor, pad: Sequence[int], mode: str = ..., value: float | None = ...
) -> Tensor: ...
def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = ...,
    dropout_p: float = ...,
    is_causal: bool = ...,
    scale: float | None = ...,
    enable_gqa: bool = ...,
) -> Tensor: ...
def softplus(input: Tensor, beta: float = ..., threshold: float = ...) -> Tensor: ...
def softshrink(input: Tensor, lambd: float = ...) -> Tensor: ...
def mkldnn_linear(input: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor: ...
def mkldnn_reorder_conv2d_weight(
    self: Tensor, padding: list, stride: list, dilatation: list, groups: int
) -> Tensor: ...
def mkldnn_reorder_conv3d_weight(
    self: Tensor, padding: list, stride: list, dilatation: list, groups: int
) -> Tensor: ...
def mkldnn_prelu(input: Tensor, weight: Tensor) -> Tensor: ...
def pad_sequence(
    sequences: list[Tensor] | tuple[Tensor, ...],
    batch_first: bool = ...,
    padding_value: float = ...,
    padding_side: Literal["left", "right"] = ...,
) -> Tensor: ...
def upsample_nearest1d(
    input: Tensor,
    output_size: Sequence[int] | None,
    scale_factors: Sequence[float] | None,
) -> Tensor: ...
def upsample_nearest2d(
    input: Tensor,
    output_size: Sequence[int] | None,
    scale_factors: Sequence[float] | None,
) -> Tensor: ...
def upsample_nearest3d(
    input: Tensor,
    output_size: Sequence[int] | None,
    scale_factors: Sequence[float] | None,
) -> Tensor: ...
def upsample_linear1d(
    input: Tensor,
    output_size: Sequence[int] | None,
    align_corners: bool,
    scale_factors: Sequence[float] | None,
) -> Tensor: ...
def upsample_bilinear2d(
    input: Tensor,
    output_size: Sequence[int] | None,
    align_corners: bool,
    scale_factors: Sequence[float] | None,
) -> Tensor: ...
def upsample_trilinear3d(
    input: Tensor,
    output_size: Sequence[int] | None,
    align_corners: bool,
    scale_factors: Sequence[float] | None,
) -> Tensor: ...
def upsample_bicubic2d(
    input: Tensor,
    output_size: Sequence[int] | None,
    align_corners: bool,
    scale_factors: Sequence[float] | None,
) -> Tensor: ...
def flatten_dense_tensors(tensors: list[Tensor]) -> Tensor: ...
def unflatten_dense_tensors(flat: Tensor, tensors: list[Tensor]) -> list[Tensor]: ...
