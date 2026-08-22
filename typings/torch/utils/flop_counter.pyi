from collections.abc import Callable
from types import NotImplementedType
from typing import Any, Self, TypeVar
from typing_extensions import ParamSpec

from torch import Size
from torch.utils._python_dispatch import TorchDispatchMode

import torch

__all__ = ["FlopCounterMode", "register_flop_formula"]
_T = TypeVar("_T")
_P = ParamSpec("_P")
aten = ...

def get_shape(i) -> Size: ...

flop_registry: dict[Any, Any] = ...

def shape_wrapper(f) -> _Wrapped[..., Any, ..., Any]: ...
def register_flop_formula(
    targets, get_raw=...
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
@register_flop_formula(aten.mm)
def mm_flop(a_shape, b_shape, *args, out_shape=..., **kwargs) -> int: ...
@register_flop_formula(aten.addmm)
def addmm_flop(self_shape, a_shape, b_shape, out_shape=..., **kwargs) -> int: ...
@register_flop_formula(aten.bmm)
def bmm_flop(a_shape, b_shape, out_shape=..., **kwargs) -> int: ...
@register_flop_formula(aten.baddbmm)
def baddbmm_flop(self_shape, a_shape, b_shape, out_shape=..., **kwargs) -> int: ...
def conv_flop_count(
    x_shape: list[int], w_shape: list[int], out_shape: list[int], transposed: bool = ...
) -> int: ...
@register_flop_formula(
    [
        aten.convolution,
        aten._convolution,
        aten.cudnn_convolution,
        aten._slow_conv2d_forward,
    ]
)
def conv_flop(
    x_shape,
    w_shape,
    _bias,
    _stride,
    _padding,
    _dilation,
    transposed,
    *args,
    out_shape=...,
    **kwargs,
) -> int: ...
@register_flop_formula(aten.convolution_backward)
def conv_backward_flop(
    grad_out_shape,
    x_shape,
    w_shape,
    _bias,
    _stride,
    _padding,
    _dilation,
    transposed,
    _output_padding,
    _groups,
    output_mask,
    out_shape,
) -> int: ...
def sdpa_flop_count(query_shape, key_shape, value_shape) -> int: ...
@register_flop_formula(
    [
        aten._scaled_dot_product_efficient_attention,
        aten._scaled_dot_product_flash_attention,
        aten._scaled_dot_product_cudnn_attention,
    ]
)
def sdpa_flop(
    query_shape, key_shape, value_shape, *args, out_shape=..., **kwargs
) -> int: ...
def sdpa_backward_flop_count(
    grad_out_shape, query_shape, key_shape, value_shape
) -> int: ...
@register_flop_formula(
    [
        aten._scaled_dot_product_efficient_attention_backward,
        aten._scaled_dot_product_flash_attention_backward,
        aten._scaled_dot_product_cudnn_attention_backward,
    ]
)
def sdpa_backward_flop(
    grad_out_shape, query_shape, key_shape, value_shape, *args, out_shape=..., **kwargs
) -> int: ...

flop_registry = ...

def normalize_tuple(x) -> tuple[Any] | tuple[Any, ...]: ...

suffixes = ...

def get_suffix_str(number) -> str: ...
def convert_num_with_suffix(number, suffix) -> str: ...
def convert_to_percent_str(num, denom) -> str: ...

class FlopCounterMode:
    def __init__(
        self,
        mods: torch.nn.Module | list[torch.nn.Module] | None = ...,
        depth: int = ...,
        display: bool = ...,
        custom_mapping: dict[Any, Any] | None = ...,
    ) -> None: ...
    def get_total_flops(self) -> int: ...
    def get_flop_counts(self) -> dict[str, dict[Any, int]]: ...
    def get_table(self, depth=...) -> str: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args) -> None: ...

class _FlopCounterMode(TorchDispatchMode):
    supports_higher_order_operators = ...
    def __init__(self, counter: FlopCounterMode) -> None: ...
    def __torch_dispatch__(
        self, func, types, args=..., kwargs=...
    ) -> NotImplementedType | None: ...
