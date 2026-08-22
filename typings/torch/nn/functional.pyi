from collections.abc import Callable, Sequence
from typing import Any, Literal, overload

from torch import Tensor
from torch.types import _dtype, _int, _size

from .common_types import (
    _ratio_any_t,
    _size_1_t,
    _size_2_opt_t,
    _size_2_t,
    _size_3_opt_t,
    _size_3_t,
    _size_any_t,
)

__all__ = ["GRID_SAMPLE_INTERPOLATION_MODES", "GRID_SAMPLE_PADDING_MODES"]
type GRID_SAMPLE_INTERPOLATION_MODES = dict[str, int]
type GRID_SAMPLE_PADDING_MODES = dict[str, int]
__all__ += ["_canonical_mask"]
__all__ += ["_none_or_dtype"]

def adaptive_avg_pool2d(input: Tensor, output_size: _size_2_opt_t) -> Tensor: ...

__all__ += ["adaptive_avg_pool2d"]

def adaptive_avg_pool3d(input: Tensor, output_size: _size_3_opt_t) -> Tensor: ...

__all__ += ["adaptive_avg_pool3d"]

def adaptive_max_pool1d_with_indices(
    input: Tensor, output_size: _size, return_indices: bool = ...
) -> tuple[Tensor, Tensor]: ...

__all__ += ["adaptive_max_pool1d_with_indices"]

def adaptive_max_pool2d_with_indices(
    input: Tensor, output_size: _size_2_opt_t, return_indices: bool = ...
) -> tuple[Tensor, Tensor]: ...

__all__ += ["adaptive_max_pool2d_with_indices"]

def adaptive_max_pool3d_with_indices(
    input: Tensor, output_size: _size_3_opt_t, return_indices: bool = ...
) -> tuple[Tensor, Tensor]: ...

__all__ += ["adaptive_max_pool3d_with_indices"]

def affine_grid(
    theta: Tensor, size: list[int], align_corners: Any | None = ...
) -> Tensor: ...

__all__ += ["affine_grid"]

def alpha_dropout(
    input: Tensor, p: float = ..., training: bool = ..., inplace: bool = ...
) -> Tensor: ...

__all__ += ["alpha_dropout"]

def assert_int_or_pair(arg: Any, arg_name: Any, message: Any) -> None: ...

__all__ += ["assert_int_or_pair"]

def batch_norm(
    input: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = ...,
    bias: Tensor | None = ...,
    training: bool = ...,
    momentum: float = ...,
    eps: float = ...,
) -> Tensor: ...

__all__ += ["batch_norm"]

def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = ...,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
    pos_weight: Tensor | None = ...,
) -> Tensor: ...

__all__ += ["binary_cross_entropy_with_logits"]

def binary_cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = ...,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> Tensor: ...

__all__ += ["binary_cross_entropy"]

def celu(input: Tensor, alpha: float = ..., inplace: bool = ...) -> Tensor: ...

__all__ += ["celu"]

def cosine_embedding_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = ...,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> Tensor: ...

__all__ += ["cosine_embedding_loss"]

def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = ...,
    size_average: bool | None = ...,
    ignore_index: int = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
    label_smoothing: float = ...,
) -> Tensor: ...

__all__ += ["cross_entropy"]

def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: int = ...,
    reduction: str = ...,
    zero_infinity: bool = ...,
) -> Tensor: ...

__all__ += ["ctc_loss"]

def dropout(
    input: Tensor, p: float = ..., training: bool = ..., inplace: bool = ...
) -> Tensor: ...

__all__ += ["dropout"]

def dropout1d(
    input: Tensor, p: float = ..., training: bool = ..., inplace: bool = ...
) -> Tensor: ...

__all__ += ["dropout1d"]

def dropout2d(
    input: Tensor, p: float = ..., training: bool = ..., inplace: bool = ...
) -> Tensor: ...

__all__ += ["dropout2d"]

def dropout3d(
    input: Tensor, p: float = ..., training: bool = ..., inplace: bool = ...
) -> Tensor: ...

__all__ += ["dropout3d"]

def elu(input: Tensor, alpha: float = ..., inplace: bool = ...) -> Tensor: ...

__all__ += ["elu"]

def embedding_bag(
    input: Tensor,
    weight: Tensor,
    offsets: Tensor | None = ...,
    max_norm: float | None = ...,
    norm_type: float = ...,
    scale_grad_by_freq: bool = ...,
    mode: str = ...,
    sparse: bool = ...,
    per_sample_weights: Tensor | None = ...,
    include_last_offset: bool = ...,
    padding_idx: int | None = ...,
) -> Tensor: ...

__all__ += ["embedding_bag"]

def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: int | None = ...,
    max_norm: float | None = ...,
    norm_type: float = ...,
    scale_grad_by_freq: bool = ...,
    sparse: bool = ...,
) -> Tensor: ...

__all__ += ["embedding"]

def feature_alpha_dropout(
    input: Tensor, p: float = ..., training: bool = ..., inplace: bool = ...
) -> Tensor: ...

__all__ += ["feature_alpha_dropout"]

def fold(
    input: Tensor,
    output_size: _size_any_t,
    kernel_size: _size_any_t,
    dilation: _size_any_t = ...,
    padding: _size_any_t = ...,
    stride: _size_any_t = ...,
) -> Tensor: ...

__all__ += ["fold"]

def fractional_max_pool2d_with_indices(
    input: Tensor,
    kernel_size: _size,
    output_size: _size | None = ...,
    output_ratio: _ratio_any_t | None = ...,
    return_indices: bool = ...,
    _random_samples: Tensor | None = ...,
) -> tuple[Tensor, Tensor]: ...

__all__ += ["fractional_max_pool2d_with_indices"]

def fractional_max_pool3d_with_indices(
    input: Tensor,
    kernel_size: _size,
    output_size: _size | None = ...,
    output_ratio: _ratio_any_t | None = ...,
    return_indices: bool = ...,
    _random_samples: Tensor | None = ...,
) -> tuple[Tensor, Tensor]: ...

__all__ += ["fractional_max_pool3d_with_indices"]

def gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    var: Tensor | float,
    full: bool | None = ...,
    eps: float | None = ...,
    reduction: str | None = ...,
) -> Tensor: ...

__all__ += ["gaussian_nll_loss"]

def glu(input: Tensor, dim: int = ...) -> Tensor: ...

__all__ += ["glu"]

def grid_sample(
    input: Tensor,
    grid: Tensor,
    mode: str = ...,
    padding_mode: str = ...,
    align_corners: Any | None = ...,
) -> Tensor: ...

__all__ += ["grid_sample"]

def group_norm(
    input: Tensor,
    num_groups: int,
    weight: Tensor | None = ...,
    bias: Tensor | None = ...,
    eps: float = ...,
) -> Tensor: ...

__all__ += ["group_norm"]

def gumbel_softmax(
    logits: Tensor, tau: float = ..., hard: bool = ..., eps: float = ..., dim: int = ...
) -> Tensor: ...

__all__ += ["gumbel_softmax"]

def hardsigmoid(input: Tensor, inplace: bool = ...) -> Tensor: ...

__all__ += ["hardsigmoid"]

def hardswish(input: Tensor, inplace: bool = ...) -> Tensor: ...

__all__ += ["hardswish"]

def hardtanh(
    input: Tensor, min_val: float = ..., max_val: float = ..., inplace: bool = ...
) -> Tensor: ...

__all__ += ["hardtanh"]

def hinge_embedding_loss(
    input: Tensor,
    target: Tensor,
    margin: float = ...,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> Tensor: ...

__all__ += ["hinge_embedding_loss"]

def huber_loss(
    input: Tensor, target: Tensor, reduction: str = ..., delta: float = ...
) -> Tensor: ...

__all__ += ["huber_loss"]

def instance_norm(
    input: Tensor,
    running_mean: Tensor | None = ...,
    running_var: Tensor | None = ...,
    weight: Tensor | None = ...,
    bias: Tensor | None = ...,
    use_input_stats: bool = ...,
    momentum: float = ...,
    eps: float = ...,
) -> Tensor: ...

__all__ += ["instance_norm"]

def interpolate(
    input: Tensor,
    size: int | Sequence[int] | None = ...,
    scale_factor: float | Sequence[float] | None = ...,
    mode: str = ...,
    align_corners: bool | None = ...,
    recompute_scale_factor: bool | None = ...,
    antialias: bool = ...,
) -> Tensor: ...

__all__ += ["interpolate"]

def kl_div(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
    log_target: bool = ...,
) -> Tensor: ...

__all__ += ["kl_div"]

def l1_loss(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> Tensor: ...

__all__ += ["l1_loss"]

def layer_norm(
    input: Tensor,
    normalized_shape: Sequence[int],
    weight: Tensor | None = ...,
    bias: Tensor | None = ...,
    eps: float = ...,
) -> Tensor: ...

__all__ += ["layer_norm"]

def leaky_relu(
    input: Tensor, negative_slope: float = ..., inplace: bool = ...
) -> Tensor: ...

__all__ += ["leaky_relu"]

def local_response_norm(
    input: Tensor, size: int, alpha: float = ..., beta: float = ..., k: float = ...
) -> Tensor: ...

__all__ += ["local_response_norm"]

def log_softmax(
    input: Tensor,
    dim: int | None = ...,
    _stacklevel: int = ...,
    dtype: _dtype | None = ...,
) -> Tensor: ...

__all__ += ["log_softmax"]

def lp_pool1d(
    input: Tensor,
    norm_type: float,
    kernel_size: _size_1_t,
    stride: _size | int | None = ...,
    ceil_mode: bool = ...,
) -> Tensor: ...

__all__ += ["lp_pool1d"]

def lp_pool2d(
    input: Tensor,
    norm_type: float,
    kernel_size: _size_2_t,
    stride: _size | int | None = ...,
    ceil_mode: bool = ...,
) -> Tensor: ...

__all__ += ["lp_pool2d"]

def lp_pool3d(
    input: Tensor,
    norm_type: float,
    kernel_size: _size_3_t,
    stride: _size | int | None = ...,
    ceil_mode: bool = ...,
) -> Tensor: ...

__all__ += ["lp_pool3d"]

def margin_ranking_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = ...,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> Tensor: ...

__all__ += ["margin_ranking_loss"]

def max_pool1d_with_indices(
    input: Tensor,
    kernel_size: _size,
    stride: _size | None = ...,
    padding: _size = ...,
    dilation: _size = ...,
    ceil_mode: bool = ...,
    return_indices: bool = ...,
) -> tuple[Tensor, Tensor]: ...

__all__ += ["max_pool1d_with_indices"]

def max_pool2d_with_indices(
    input: Tensor,
    kernel_size: _size,
    stride: _size | None = ...,
    padding: _size = ...,
    dilation: _size = ...,
    ceil_mode: bool = ...,
    return_indices: bool = ...,
) -> tuple[Tensor, Tensor]: ...

__all__ += ["max_pool2d_with_indices"]

def max_pool3d_with_indices(
    input: Tensor,
    kernel_size: _size,
    stride: _size | None = ...,
    padding: _size = ...,
    dilation: _size = ...,
    ceil_mode: bool = ...,
    return_indices: bool = ...,
) -> tuple[Tensor, Tensor]: ...

__all__ += ["max_pool3d_with_indices"]

def max_unpool1d(
    input: Tensor,
    indices: Tensor,
    kernel_size: _size,
    stride: _size | None = ...,
    padding: _size = ...,
    output_size: _size | None = ...,
) -> Tensor: ...

__all__ += ["max_unpool1d"]

def max_unpool2d(
    input: Tensor,
    indices: Tensor,
    kernel_size: _size,
    stride: _size | None = ...,
    padding: _size = ...,
    output_size: _size | None = ...,
) -> Tensor: ...

__all__ += ["max_unpool2d"]

def max_unpool3d(
    input: Tensor,
    indices: Tensor,
    kernel_size: _size,
    stride: _size | None = ...,
    padding: _size = ...,
    output_size: _size | None = ...,
) -> Tensor: ...

__all__ += ["max_unpool3d"]

def mish(input: Tensor, inplace: bool = ...) -> Tensor: ...

__all__ += ["mish"]

def mse_loss(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> Tensor: ...

__all__ += ["mse_loss"]

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor | None,
    in_proj_bias: Tensor | None,
    bias_k: Tensor | None,
    bias_v: Tensor | None,
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor | None,
    training: bool = ...,
    key_padding_mask: Tensor | None = ...,
    need_weights: bool = ...,
    attn_mask: Tensor | None = ...,
    use_separate_proj_weight: bool = ...,
    q_proj_weight: Tensor | None = ...,
    k_proj_weight: Tensor | None = ...,
    v_proj_weight: Tensor | None = ...,
    static_k: Tensor | None = ...,
    static_v: Tensor | None = ...,
    average_attn_weights: bool = ...,
    is_causal: bool = ...,
) -> tuple[Tensor, Tensor | None]: ...

__all__ += ["multi_head_attention_forward"]

def multi_margin_loss(
    input: Tensor,
    target: Tensor,
    p: int = ...,
    margin: float = ...,
    weight: Tensor | None = ...,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> Tensor: ...

__all__ += ["multi_margin_loss"]

def multilabel_margin_loss(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> Tensor: ...

__all__ += ["multilabel_margin_loss"]

def multilabel_soft_margin_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = ...,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> Tensor: ...

__all__ += ["multilabel_soft_margin_loss"]

def nll_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = ...,
    size_average: bool | None = ...,
    ignore_index: int = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> Tensor: ...

__all__ += ["nll_loss"]

def normalize(
    input: Tensor,
    p: float = ...,
    dim: int = ...,
    eps: float = ...,
    out: Tensor | None = ...,
) -> Tensor: ...

__all__ += ["normalize"]

def poisson_nll_loss(
    input: Tensor,
    target: Tensor,
    log_input: bool = ...,
    full: bool = ...,
    size_average: bool | None = ...,
    eps: float = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> Tensor: ...

__all__ += ["poisson_nll_loss"]

def relu(input: Tensor, inplace: bool = ...) -> Tensor: ...

__all__ += ["relu"]

def relu6(input: Tensor, inplace: bool = ...) -> Tensor: ...

__all__ += ["relu6"]

def rms_norm(
    input: Tensor,
    normalized_shape: Sequence[int],
    weight: Tensor | None = ...,
    eps: float | None = ...,
) -> Tensor: ...

__all__ += ["rms_norm"]

def rrelu(
    input: Tensor,
    lower: float = ...,
    upper: float = ...,
    training: bool = ...,
    inplace: bool = ...,
) -> Tensor: ...

__all__ += ["rrelu"]

def selu(input: Tensor, inplace: bool = ...) -> Tensor: ...

__all__ += ["selu"]

def sigmoid(input: Any) -> Tensor: ...

__all__ += ["sigmoid"]

def silu(input: Tensor, inplace: bool = ...) -> Tensor: ...

__all__ += ["silu"]

def smooth_l1_loss(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
    beta: float = ...,
) -> Tensor: ...

__all__ += ["smooth_l1_loss"]

def soft_margin_loss(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> Tensor: ...

__all__ += ["soft_margin_loss"]

def softmax(
    input: Tensor,
    dim: int | None = ...,
    _stacklevel: int = ...,
    dtype: _dtype | None = ...,
) -> Tensor: ...

__all__ += ["softmax"]

def softmin(
    input: Tensor,
    dim: int | None = ...,
    _stacklevel: int = ...,
    dtype: _dtype | None = ...,
) -> Tensor: ...

__all__ += ["softmin"]

def softsign(input: Any): ...

__all__ += ["softsign"]

def tanh(input: Any): ...

__all__ += ["tanh"]

def tanhshrink(input: Any): ...

__all__ += ["tanhshrink"]

def threshold(
    input: Tensor, threshold: float, value: float, inplace: bool = ...
) -> Tensor: ...

__all__ += ["threshold"]

def triplet_margin_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = ...,
    p: float = ...,
    eps: float = ...,
    swap: bool = ...,
    size_average: bool | None = ...,
    reduce: bool | None = ...,
    reduction: str = ...,
) -> Tensor: ...

__all__ += ["triplet_margin_loss"]

def triplet_margin_with_distance_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    *,
    distance_function: Callable[[Tensor, Tensor], Tensor] | None = ...,
    margin: float = ...,
    swap: bool = ...,
    reduction: str = ...,
) -> Tensor: ...

__all__ += ["triplet_margin_with_distance_loss"]

def unfold(
    input: Tensor,
    kernel_size: _size_any_t,
    dilation: _size_any_t = ...,
    padding: _size_any_t = ...,
    stride: _size_any_t = ...,
) -> Tensor: ...

__all__ += ["unfold"]

def upsample_bilinear(
    input: Any, size: Any | None = ..., scale_factor: Any | None = ...
): ...

__all__ += ["upsample_bilinear"]

def upsample_nearest(
    input: Any, size: Any | None = ..., scale_factor: Any | None = ...
): ...

__all__ += ["upsample_nearest"]

def upsample(
    input: Any,
    size: Any | None = ...,
    scale_factor: Any | None = ...,
    mode: str = ...,
    align_corners: Any | None = ...,
): ...

__all__ += ["upsample"]

@overload
def adaptive_max_pool1d(
    input: Tensor, output_size: _int | _size, return_indices: Literal[False] = ...
) -> Tensor: ...
@overload
def adaptive_max_pool1d(
    input: Tensor, output_size: _int | _size, return_indices: Literal[True], /
) -> tuple[Tensor, Tensor]: ...
@overload
def adaptive_max_pool1d(
    input: Tensor, output_size: _int | _size, *, return_indices: Literal[True]
) -> tuple[Tensor, Tensor]: ...
@overload
def adaptive_max_pool2d(
    input: Tensor, output_size: _int | _size, return_indices: Literal[False] = ...
) -> Tensor: ...
@overload
def adaptive_max_pool2d(
    input: Tensor, output_size: _int | _size, return_indices: Literal[True], /
) -> tuple[Tensor, Tensor]: ...
@overload
def adaptive_max_pool2d(
    input: Tensor, output_size: _int | _size, *, return_indices: Literal[True]
) -> tuple[Tensor, Tensor]: ...
@overload
def adaptive_max_pool3d(
    input: Tensor, output_size: _int | _size, return_indices: Literal[False] = ...
) -> Tensor: ...
@overload
def adaptive_max_pool3d(
    input: Tensor, output_size: _int | _size, return_indices: Literal[True], /
) -> tuple[Tensor, Tensor]: ...
@overload
def adaptive_max_pool3d(
    input: Tensor, output_size: _int | _size, *, return_indices: Literal[True]
) -> tuple[Tensor, Tensor]: ...
@overload
def fractional_max_pool2d(
    input: Tensor,
    kernel_size: _int | _size,
    output_size: _int | _size | None = ...,
    output_ratio: _ratio_any_t | None = ...,
    return_indices: Literal[False] = ...,
    _random_samples: Tensor | None = ...,
) -> Tensor: ...
@overload
def fractional_max_pool2d(
    input: Tensor,
    kernel_size: _int | _size,
    output_size: _int | _size | None,
    output_ratio: _ratio_any_t | None,
    return_indices: Literal[True],
    /,
    _random_samples: Tensor | None = ...,
) -> tuple[Tensor, Tensor]: ...
@overload
def fractional_max_pool2d(
    input: Tensor,
    kernel_size: _int | _size,
    output_size: _int | _size | None = ...,
    output_ratio: _ratio_any_t | None = ...,
    *,
    return_indices: Literal[True],
    _random_samples: Tensor | None = ...,
) -> tuple[Tensor, Tensor]: ...
@overload
def fractional_max_pool3d(
    input: Tensor,
    kernel_size: _int | _size,
    output_size: _int | _size | None = ...,
    output_ratio: _ratio_any_t | None = ...,
    return_indices: Literal[False] = ...,
    _random_samples: Tensor | None = ...,
) -> Tensor: ...
@overload
def fractional_max_pool3d(
    input: Tensor,
    kernel_size: _int | _size,
    output_size: _int | _size | None,
    output_ratio: _ratio_any_t | None,
    return_indices: Literal[True],
    /,
    _random_samples: Tensor | None = ...,
) -> tuple[Tensor, Tensor]: ...
@overload
def fractional_max_pool3d(
    input: Tensor,
    kernel_size: _int | _size,
    output_size: _int | _size | None = ...,
    output_ratio: _ratio_any_t | None = ...,
    *,
    return_indices: Literal[True],
    _random_samples: Tensor | None = ...,
) -> tuple[Tensor, Tensor]: ...
@overload
def max_pool1d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: bool = ...,
    return_indices: Literal[False] = ...,
) -> Tensor: ...
@overload
def max_pool1d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None,
    padding: _int | _size,
    dilation: _int | _size,
    ceil_mode: bool,
    return_indices: Literal[True],
    /,
) -> tuple[Tensor, Tensor]: ...
@overload
def max_pool1d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: bool = ...,
    *,
    return_indices: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def max_pool2d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: bool = ...,
    return_indices: Literal[False] = ...,
) -> Tensor: ...
@overload
def max_pool2d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None,
    padding: _int | _size,
    dilation: _int | _size,
    ceil_mode: bool,
    return_indices: Literal[True],
    /,
) -> tuple[Tensor, Tensor]: ...
@overload
def max_pool2d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: bool = ...,
    *,
    return_indices: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def max_pool3d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: bool = ...,
    return_indices: Literal[False] = ...,
) -> Tensor: ...
@overload
def max_pool3d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None,
    padding: _int | _size,
    dilation: _int | _size,
    ceil_mode: bool,
    return_indices: Literal[True],
    /,
) -> tuple[Tensor, Tensor]: ...
@overload
def max_pool3d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: bool = ...,
    *,
    return_indices: Literal[True],
) -> tuple[Tensor, Tensor]: ...

__all__ += [
    "adaptive_avg_pool1d",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "bilinear",
    "celu_",
    "channel_shuffle",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_tbc",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "cosine_similarity",
    "elu_",
    "fractional_max_pool2d",
    "fractional_max_pool3d",
    "gelu",
    "hardshrink",
    "hardtanh_",
    "leaky_relu_",
    "linear",
    "logsigmoid",
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "native_channel_shuffle",
    "one_hot",
    "pad",
    "pairwise_distance",
    "pdist",
    "pixel_shuffle",
    "pixel_unshuffle",
    "prelu",
    "relu_",
    "rrelu_",
    "scaled_dot_product_attention",
    "selu_",
    "softplus",
    "softshrink",
]

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
def pad(
    input: Tensor,
    pad: Sequence[int],
    mode: str = ...,
    value: float | None = ...,
) -> Tensor: ...
def gelu(input: Tensor, approximate: str = ...) -> Tensor: ...
def softplus(input: Tensor, beta: float = ..., threshold: float = ...) -> Tensor: ...
def logsigmoid(input: Tensor) -> Tensor: ...
def cosine_similarity(
    x1: Tensor, x2: Tensor, dim: int = ..., eps: float = ...
) -> Tensor: ...
def avg_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None = ...,
    padding: int | tuple[int, ...] = ...,
    ceil_mode: bool = ...,
    count_include_pad: bool = ...,
    divisor_override: int | None = ...,
) -> Tensor: ...
def avg_pool3d(
    input: Tensor,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...] | None = ...,
    padding: int | tuple[int, ...] = ...,
    ceil_mode: bool = ...,
    count_include_pad: bool = ...,
    divisor_override: int | None = ...,
) -> Tensor: ...
def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = ...,
    stride: int | tuple[int] = ...,
    padding: int | tuple[int] | str = ...,
    dilation: int | tuple[int] = ...,
    groups: int = ...,
) -> Tensor: ...
def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = ...,
    stride: int | tuple[int, int] = ...,
    padding: int | tuple[int, int] | str = ...,
    dilation: int | tuple[int, int] = ...,
    groups: int = ...,
) -> Tensor: ...
def conv3d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = ...,
    stride: int | tuple[int, int, int] = ...,
    padding: int | tuple[int, int, int] | str = ...,
    dilation: int | tuple[int, int, int] = ...,
    groups: int = ...,
) -> Tensor: ...
def one_hot(tensor: Tensor, num_classes: int = ...) -> Tensor: ...
def linear(input: Tensor, weight: Tensor, bias: Tensor | None = ...) -> Tensor: ...
