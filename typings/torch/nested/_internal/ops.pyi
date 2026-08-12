from collections.abc import Callable
from torch import Tensor, layout
from typing import Any
from typing import *

from torch._higher_order_ops.flex_attention import (
    flex_attention as flex_attention_hop,
    flex_attention_backward as flex_attention_backward_hop,
)
from torch.fx.graph_module import GraphModule

import torch

from .nested_tensor import NestedTensor

__all__: list[Any] = ...
JAGGED_OPS_TABLE: Dict[Any, Any] = ...

def check_schema(schema_str: str, func, *args, **kwargs) -> None: ...
def check_ragged_dim_same(
    func, a: NestedTensor, a_name: str, b: NestedTensor, b_name: str
) -> None: ...
def raggedness_matches(nt, size) -> bool: ...
def squeeze_leading_ones(t): ...
def register_func(tables, aten_ops, schema_str) -> Callable[..., Any]: ...

register_jagged_func = ...

def lookup_jagged(func, *args, **kwargs) -> Callable | None: ...
def extract_kwargs(arg) -> dict[str, Any]: ...
def jagged_unary_pointwise(func, *args, **kwargs) -> NestedTensor: ...
def jagged_binary_pointwise(func, *args, **kwargs) -> NestedTensor: ...
def jagged_torch_function(func, *args, **kwargs) -> Tensor | Any: ...
@register_jagged_func(
    [
        torch.ops.aten.is_non_overlapping_and_dense.default,
        torch.ops.aten.sym_size.default,
        torch.ops.aten.dim.default,
        torch.ops.aten.numel.default,
        torch.ops.aten.sym_numel.default,
        torch.ops.aten.sym_stride.default,
        torch.ops.aten.sym_storage_offset.default,
    ],
    "self: jt_all",
)
def tensor_attr_supported_getter(func, *args, **kwargs) -> int | None: ...
@register_jagged_func(torch.ops.prim.layout.default, "self: jt_all")
def prim_layout_default(func, *args, **kwargs) -> layout: ...
@register_jagged_func([torch.ops.aten.size.default], "self: jt_all")
def tensor_attr_unsupported_getter(func, *args, **kwargs) -> None: ...
@register_jagged_func(torch.ops.aten.is_contiguous.default, "self: jt_all")
def is_contiguous_general(func, *args, **kwargs) -> bool: ...
@register_jagged_func(
    torch.ops.aten.sym_is_contiguous.default, "self: jt_all, memory_format: any?"
)
def sym_is_contiguous_general(func, *args, **kwargs) -> bool: ...
@register_jagged_func(
    torch.ops.aten.clone.default, "input: jt_all, memory_format: any?"
)
def clone_default(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.linear.default, "input: jt, weight: t, bias: t?")
def linear_default(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.linear_backward.default, ...)
def linear_backward_default(
    func, *args, **kwargs
) -> tuple[NestedTensor | None, Tensor | None, Any | Tensor | None]: ...
@register_jagged_func(torch.ops.aten.to.dtype, "input: jt_all, dtype: any")
def to_dtype(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten._to_copy.default, "self: jt_all")
def to_copy_default(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(
    torch.ops.aten.copy_.default, "self: jt_all, src: jt_all, non_blocking: any?"
)
def copy_default(func, *args, **kwargs) -> Any: ...
@register_jagged_func(
    [
        torch.ops.aten.empty_like.default,
        torch.ops.aten.ones_like.default,
        torch.ops.aten.zeros_like.default,
        torch.ops.aten.rand_like.default,
        torch.ops.aten.randn_like.default,
    ],
    "self: jt_all",
)
def like_factory_default(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.zero_.default, "self: jt_all")
def zero__default(func, *args, **kwargs) -> Any: ...
@register_jagged_func(
    torch.ops.aten.native_dropout.default, "self: jt, float: any, train: any?"
)
def native_dropout_default(
    func, *args, **kwargs
) -> tuple[NestedTensor, NestedTensor]: ...
@register_jagged_func(
    torch.ops.aten.native_dropout_backward.default,
    "grad_output: jt, mask: jt, scale: any",
)
def native_dropout_backward_default(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(
    torch.ops.aten.prod.dim_int, "self: jt_all, dim: any, keepdim: any?, dtype: any?"
)
def prod_dim_int(func, *args, **kwargs) -> list[Any] | tuple[Any, ...] | PyTree: ...
@register_jagged_func(torch.ops.aten.prod.default, "self: jt_all, dtype: any?")
def prod_default(func, *args, **kwargs): ...
@register_jagged_func(
    torch.ops.aten.split.Tensor, "self: jt, split_size: any, dim: any?"
)
def split_tensor(func, *args, **kwargs) -> tuple[NestedTensor, ...]: ...
@register_jagged_func(
    torch.ops.aten.split_with_sizes.default, "self: jt, split_sizes: any, dim: any?"
)
def split_with_sizes_default(func, *args, **kwargs) -> list[NestedTensor]: ...
@register_jagged_func(
    torch.ops.aten.narrow.default, "self: jt, dim: any, start: any, length: any"
)
def narrow(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.chunk.default, "self: jt, chunks: any, dim: any?")
def chunk_default(func, *args, **kwargs) -> list[NestedTensor]: ...
@register_jagged_func(torch.ops.aten.unbind.int, "self: jt_all, dim: any?")
def unbind_int(func, *args, **kwargs) -> tuple[Tensor, ...] | list[Tensor]: ...
@register_jagged_func(torch.ops.aten.squeeze.dim, "self: jt, dim: any")
def squeeze_dim(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.unsqueeze.default, "self: jt_all, dim: any")
def unsqueeze_default(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.cat.default, "tensors: any, dim: any")
def cat_default(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.matmul.default, "self: any, other: any")
def matmul_default(func, *args, **kwargs) -> Tensor | NestedTensor: ...
@register_jagged_func(torch.ops.aten.bmm.default, "self: jt_all, mat2: any")
def bmm_default(func, *args, **kwargs) -> Tensor | NestedTensor: ...
@register_jagged_func(
    torch.ops.aten.expand.default, "self: jt_all, size: any, implicit: any?"
)
def expand_default(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.expand_as.default, "self: t, other: jt")
def expand_as_default(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.broadcast_to.default, "self: jt_all, size: any")
def broadcast_to(func, *args, **kwargs) -> Any: ...
@register_jagged_func(torch.ops.aten.broadcast_tensors.default, "tensors: any")
def broadcast_tensors(func, *args, **kwargs) -> tuple[Any, ...]: ...
@register_jagged_func(
    torch.ops.aten.where.self, "condition: jt_all, self: any, other: any"
)
def where_self(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.is_pinned.default, "self: jt, device: any?")
def is_pinned_default(func, *args, **kwargs): ...
@register_jagged_func(
    torch.ops.aten.is_same_size.default, "self: jt_all, other: jt_all"
)
def is_same_size_default(func, *args, **kwargs): ...
@register_jagged_func(torch.ops.aten.sum.default, "self: jt_all, dtype: any?")
def sum_default(func, *args, **kwargs): ...
@register_jagged_func(torch.ops.aten.sum.dim_IntList, ...)
def sum_dim_IntList(func, *args, **kwargs) -> list[Any] | tuple[Any, ...] | PyTree: ...
@register_jagged_func(
    torch.ops.aten.transpose.int, "self: jt_all, dim0: any, dim1: any"
)
def transpose_int(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.permute.default, "self: jt_all, dims: any")
def permute_default(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(
    [torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default],
    "self: jt_all, size: any",
)
def view_default(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.native_layer_norm.default, ...)
def native_layer_norm_default(
    func, *args, **kwargs
) -> tuple[NestedTensor, Any, Tensor] | tuple[NestedTensor, Any, Any]: ...
@register_jagged_func(torch.ops.aten.native_layer_norm_backward.default, ...)
def native_layer_norm_backward_default(
    func, *args, **kwargs
) -> tuple[None, Any, Any] | tuple[NestedTensor, Any, Any]: ...
@register_jagged_func(torch.ops.aten.select.int, "self: jt_all, dim: any, index: any")
def select_int(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.slice.Tensor, ...)
def slice_tensor(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.index_put.default, ...)
@register_jagged_func(torch.ops.aten.index_put_.default, ...)
def index_put_(func, *args, **kwargs) -> NestedTensor | Tensor: ...
@register_jagged_func(torch.ops.aten.convolution.default, ...)
def convolution_default(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.mean.dim, ...)
def mean_dim(func, *args, **kwargs) -> list[Any] | tuple[Any, ...]: ...
@register_jagged_func(torch.ops.aten.mean.default, "self: jt_all, dtype: any?")
def mean_default(func, *args, **kwargs): ...
@register_jagged_func(torch.ops.aten.any.dims, "self: jt_all, dim: any?, keepdim: any?")
def any_dims(func, *args, **kwargs) -> list[Any] | tuple[Any, ...] | PyTree: ...
@register_jagged_func(torch.ops.aten.any.dim, "self: jt_all, dim: any, keepdim: any?")
def any_dim(func, *args, **kwargs) -> list[Any] | tuple[Any, ...] | PyTree: ...
@register_jagged_func(torch.ops.aten.all.dims, "self: jt_all, dim: any?, keepdim: any?")
def all_dims(func, *args, **kwargs) -> list[Any] | tuple[Any, ...] | PyTree: ...
@register_jagged_func(torch.ops.aten.all.dim, "self: jt_all, dim: any, keepdim: any?")
def all_dim(func, *args, **kwargs) -> list[Any] | tuple[Any, ...] | PyTree: ...
@register_jagged_func(
    [
        torch.ops.aten.all.default,
        torch.ops.aten.any.default,
        torch.ops.aten.max.default,
        torch.ops.aten.min.default,
    ],
    "self: jt_all",
)
def all_any_max_min_default(func, *args, **kwargs): ...
@register_jagged_func(torch.ops.aten.min.dim, "self: jt_all, dim: any, keepdim: any?")
def min_dim(func, *args, **kwargs) -> list[Any] | tuple[Any, ...] | PyTree: ...
@register_jagged_func(torch.ops.aten.max.dim, "self: jt_all, dim: any, keepdim: any?")
def max_dim(func, *args, **kwargs) -> list[Any] | tuple[Any, ...] | PyTree: ...
@register_jagged_func(
    torch.ops.aten.amin.default, "self: jt_all, dim: any?, keepdim: any?"
)
def amin_default(func, *args, **kwargs) -> list[Any] | tuple[Any, ...] | PyTree: ...
@register_jagged_func(
    torch.ops.aten.amax.default, "self: jt_all, dim: any?, keepdim: any?"
)
def amax_default(func, *args, **kwargs) -> list[Any] | tuple[Any, ...] | PyTree: ...
@register_jagged_func(
    torch.ops.aten.argmin.default, "self: jt_all, dim: any?, keepdim: any?"
)
def argmin_default(func, *args, **kwargs) -> list[Any] | tuple[Any, ...] | PyTree: ...
@register_jagged_func(
    torch.ops.aten.argmax.default, "self: jt_all, dim: any?, keepdim: any?"
)
def argmax_default(func, *args, **kwargs) -> list[Any] | tuple[Any, ...] | PyTree: ...
@register_jagged_func(torch.ops.aten.value_selecting_reduction_backward.default, ...)
def value_selecting_reduction_backward_default(
    func, *args, **kwargs
) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.stack.default, "tensors: any, dim: any")
def stack_default(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.embedding.default, ...)
def embedding_default(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.embedding_dense_backward.default, ...)
def embedding_dense_backward_default(func, *args, **kwargs): ...
@register_jagged_func(
    [torch.ops.aten.values.default, torch.ops.aten._nested_get_values.default],
    "self: jt_all",
)
def values_default(func, *args, **kwargs) -> Any: ...
@register_jagged_func(torch.ops.aten.all.default, "self: jt_all")
def all_default(func, *args, **kwargs): ...
@register_jagged_func(
    torch.ops.aten.to_padded_tensor.default,
    "self: jt_all, padding: any, output_size: any?",
)
def to_padded_tensor_default(func, *args, **kwargs) -> Any: ...
@register_jagged_func(torch.ops.aten.masked_select.default, "self: jt, mask: any")
def masked_select_default(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.record_stream.default, "self: jt_all, s: any")
def record_stream_default(func, *args, **kwargs) -> None: ...
@register_jagged_func(
    [
        torch.ops.aten.new_empty.default,
        torch.ops.aten.new_zeros.default,
        torch.ops.aten.new_ones.default,
    ],
    ...,
)
def new_empty_default(func, *args, **kwargs): ...
@register_jagged_func(
    [
        torch.ops.aten.elu_backward.default,
        torch.ops.aten.hardshrink_backward.default,
        torch.ops.aten.hardsigmoid_backward.default,
        torch.ops.aten.hardtanh_backward.default,
        torch.ops.aten.softplus_backward.default,
        torch.ops.aten.softshrink_backward.default,
    ],
    "self: jt_all, ...",
)
def activation_backward(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.fill.Scalar, "self: jt_all, value: any")
def fill_Scalar(func, *args, **kwargs) -> NestedTensor: ...
@register_jagged_func(torch.ops.aten.fill_.Scalar, "self: jt_all, value: any")
def fill__Scalar(func, *args, **kwargs) -> Any: ...
@register_jagged_func(torch.ops.aten.frexp.Tensor, "self: jt_all")
def frexp_Tensor(func, *args, **kwargs) -> tuple[NestedTensor, NestedTensor]: ...
@register_jagged_func(
    torch.ops.aten.matmul_backward.default,
    "grad: any, self: any, other: any, mask: any",
)
def matmul_backward_default(
    func, *args, **kwargs
) -> tuple[None, None] | tuple[Tensor | None, Tensor | None]: ...
@flex_attention_hop.py_impl(NestedTensor)
def flex_njt(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    score_mod_other_buffers: Tuple = ...,
    mask_mod_other_buffers: Tuple = ...,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
@flex_attention_backward_hop.py_impl(NestedTensor)
def flex_njt_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Union[Callable, GraphModule],
    joint_graph: GraphModule,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    score_mod_other_buffers: Tuple = ...,
    mask_mod_other_buffers: Tuple = ...,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor | None, ...]
]: ...
