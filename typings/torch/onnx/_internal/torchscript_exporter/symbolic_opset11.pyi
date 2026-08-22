from collections.abc import Sequence
from typing import Any

from torch import _C
from torch.onnx._internal.torchscript_exporter import jit_utils, symbolic_helper

import torch

"""This file exports ONNX ops for opset 11."""
__all__ = [
    "Delete",
    "add",
    "append",
    "arange",
    "argsort",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "cat",
    "chunk",
    "clamp",
    "clamp_max",
    "clamp_min",
    "constant_pad_nd",
    "cumsum",
    "embedding_bag",
    "embedding_renorm",
    "flatten",
    "gather",
    "hardtanh",
    "hstack",
    "im2col",
    "index",
    "index_copy",
    "index_fill",
    "index_put",
    "insert",
    "linalg_det",
    "linalg_vector_norm",
    "logdet",
    "masked_scatter",
    "masked_select",
    "mm",
    "narrow",
    "normal",
    "pad",
    "pixel_shuffle",
    "pop",
    "prim_constant_chunk",
    "reflection_pad",
    "relu6",
    "remainder",
    "replication_pad",
    "round",
    "scatter",
    "select",
    "size",
    "sort",
    "split",
    "split_with_sizes",
    "squeeze",
    "stack",
    "topk",
    "unbind",
    "unique_dim",
    "unsqueeze",
    "vstack",
]
_onnx_symbolic = ...

@_onnx_symbolic("aten::hardtanh")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "f", "f")
def hardtanh(
    g: jit_utils.GraphContext, self: _C.Value, min_val: float, max_val: float
): ...
@_onnx_symbolic("aten::clamp")
def clamp(g: jit_utils.GraphContext, self, min, max): ...
@_onnx_symbolic("aten::clamp_min")
@symbolic_helper.parse_args("v", "v")
def clamp_min(g: jit_utils.GraphContext, self, min): ...
@_onnx_symbolic("aten::clamp_max")
@symbolic_helper.parse_args("v", "v")
def clamp_max(g: jit_utils.GraphContext, self, max): ...
@_onnx_symbolic("aten::relu6")
def relu6(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::select")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "i", "v")
def select(g: jit_utils.GraphContext, self, dim, index): ...
@_onnx_symbolic("aten::index_put")
def index_put(
    g: jit_utils.GraphContext, self, indices_list_value, values, accumulate=...
) -> None: ...
@_onnx_symbolic("aten::pixel_shuffle")
@symbolic_helper.parse_args("v", "i")
def pixel_shuffle(g: jit_utils.GraphContext, self, upscale_factor) -> None: ...
@_onnx_symbolic("aten::gather")
@symbolic_helper.parse_args("v", "i", "v", "v")
def gather(g: jit_utils.GraphContext, self, dim, index, sparse_grad=...) -> None: ...
@_onnx_symbolic("aten::scatter")
@symbolic_helper.parse_args("v", "i", "v", "v")
def scatter(g: jit_utils.GraphContext, self, dim, index, src): ...
@_onnx_symbolic("aten::cumsum")
@symbolic_helper.parse_args("v", "i", "none")
def cumsum(g: jit_utils.GraphContext, self, dim, dtype=...): ...
@_onnx_symbolic("aten::masked_select")
def masked_select(g: jit_utils.GraphContext, self, mask): ...
@_onnx_symbolic("aten::masked_scatter")
def masked_scatter(g: jit_utils.GraphContext, self, mask, source): ...
@_onnx_symbolic("aten::append")
def append(g: jit_utils.GraphContext, self, tensor): ...
@_onnx_symbolic("aten::add")
def add(g: jit_utils.GraphContext, self, other, alpha=...) -> None: ...
@_onnx_symbolic("aten::insert")
def insert(g: jit_utils.GraphContext, self, pos, tensor): ...
@_onnx_symbolic("aten::pop")
def pop(g: jit_utils.GraphContext, tensor_list, dim): ...
@_onnx_symbolic("aten::Delete")
def Delete(g: jit_utils.GraphContext, tensor_list, dim): ...
@_onnx_symbolic("aten::cat")
@symbolic_helper.quantized_args(True)
def cat(g: jit_utils.GraphContext, tensor_list, dim): ...
@_onnx_symbolic("aten::stack")
def stack(g: jit_utils.GraphContext, tensor_list, dim): ...
@_onnx_symbolic("aten::unique_dim")
@symbolic_helper.parse_args("v", "i", "i", "i", "i")
def unique_dim(
    g: jit_utils.GraphContext, self, dim, sorted, return_inverse, return_counts
) -> tuple[Any, Any, Any]: ...
@_onnx_symbolic("aten::topk")
@symbolic_helper.parse_args("v", "v", "i", "i", "i", "none")
def topk(g: jit_utils.GraphContext, self, k, dim, largest, sorted, out=...): ...
@_onnx_symbolic("aten::sort")
@symbolic_helper.parse_args("v", "i", "i", "none")
def sort(g: jit_utils.GraphContext, self, dim, descending, out=...): ...
@_onnx_symbolic("aten::argsort")
@symbolic_helper.parse_args("v", "i", "i", "none")
def argsort(g: jit_utils.GraphContext, self, dim, descending, out=...): ...
@_onnx_symbolic("aten::round")
@symbolic_helper.parse_args("v", "i")
def round(g: jit_utils.GraphContext, self, decimals=...): ...
@_onnx_symbolic("aten::remainder")
def remainder(g: jit_utils.GraphContext, input, other): ...
@_onnx_symbolic("aten::split")
@symbolic_helper.parse_args("v", "v", "i", "i")
def split(
    g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=...
) -> list[Any]: ...
@_onnx_symbolic("aten::split_with_sizes")
@symbolic_helper.parse_args("v", "v", "i", "i")
def split_with_sizes(
    g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=...
) -> list[Any]: ...
@_onnx_symbolic("aten::unbind")
@symbolic_helper.parse_args("v", "i", "i")
def unbind(g: jit_utils.GraphContext, self, dim=..., _outputs=...) -> list[Any]: ...
@_onnx_symbolic("aten::constant_pad_nd")
def constant_pad_nd(g: jit_utils.GraphContext, input, padding, value=...): ...
@_onnx_symbolic("aten::reflection_pad1d")
@_onnx_symbolic("aten::reflection_pad2d")
@_onnx_symbolic("aten::reflection_pad3d")
def reflection_pad(g: jit_utils.GraphContext, input, padding): ...
@_onnx_symbolic("aten::replication_pad1d")
@_onnx_symbolic("aten::replication_pad2d")
@_onnx_symbolic("aten::replication_pad3d")
def replication_pad(g: jit_utils.GraphContext, input, padding): ...
@_onnx_symbolic("aten::pad")
def pad(
    g: jit_utils.GraphContext,
    input: _C.Value,
    pad: _C.Value,
    mode: _C.Value,
    value: _C.Value,
) -> Value: ...
@_onnx_symbolic("aten::linalg_det")
def linalg_det(g: jit_utils.GraphContext, self): ...
@_onnx_symbolic("aten::logdet")
def logdet(g: jit_utils.GraphContext, input): ...
@_onnx_symbolic("aten::arange")
def arange(g: jit_utils.GraphContext, *args) -> None: ...
@_onnx_symbolic("aten::size")
@symbolic_helper.quantized_args(True, quantize_output=False)
def size(g: jit_utils.GraphContext, self, dim=...): ...
@_onnx_symbolic("aten::squeeze")
def squeeze(g: jit_utils.GraphContext, self, dim=...) -> Any: ...
@_onnx_symbolic("aten::unsqueeze")
def unsqueeze(g: jit_utils.GraphContext, self, dim): ...
@_onnx_symbolic("aten::mm")
def mm(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::index")
def index(g: jit_utils.GraphContext, self, index) -> None: ...
@_onnx_symbolic("aten::index_fill")
def index_fill(g: jit_utils.GraphContext, self, dim, index, value): ...
@_onnx_symbolic("aten::index_copy")
def index_copy(g: jit_utils.GraphContext, self, dim, index, source): ...
@_onnx_symbolic("aten::im2col")
@symbolic_helper.parse_args("v", "is", "is", "is", "is")
def im2col(
    g: jit_utils.GraphContext, input, kernel_size, dilation, padding, stride
): ...
@_onnx_symbolic("aten::narrow")
def narrow(g: jit_utils.GraphContext, input, dim, start, length) -> Value: ...
@_onnx_symbolic("aten::flatten")
@symbolic_helper.quantized_args(True, False, False)
@symbolic_helper.parse_args("v", "i", "i")
def flatten(g: jit_utils.GraphContext, input, start_dim, end_dim) -> None: ...
@_onnx_symbolic("aten::linalg_vector_norm")
@symbolic_helper.parse_args("v", "f", "is", "b", "v")
def linalg_vector_norm(
    g: jit_utils.GraphContext,
    self,
    ord,
    dim: Sequence[int] | None,
    keepdim: bool,
    dtype,
): ...
@_onnx_symbolic("aten::embedding_bag")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "v", "i", "i")
def embedding_bag(
    g: jit_utils.GraphContext,
    embedding_matrix,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
) -> tuple[Any, None, None, None]: ...
@_onnx_symbolic("aten::embedding_renorm")
@symbolic_helper.parse_args("v", "v", "f", "f")
def embedding_renorm(
    g: jit_utils.GraphContext, weight, indices, max_norm, norm_type
): ...
@_onnx_symbolic("aten::chunk")
def chunk(g: jit_utils.GraphContext, self, chunks, dim) -> list[Any]: ...
@_onnx_symbolic("aten::normal")
def normal(
    g: jit_utils.GraphContext,
    mean,
    std,
    sizes=...,
    generator=...,
    dtype=...,
    layout=...,
    device=...,
    pin_memory=...,
) -> None: ...
@_onnx_symbolic("aten::atleast_1d")
def atleast_1d(g: jit_utils.GraphContext, self: torch._C.Value) -> Value: ...
@_onnx_symbolic("aten::atleast_2d")
def atleast_2d(g: jit_utils.GraphContext, self: torch._C.Value) -> Value: ...
@_onnx_symbolic("aten::atleast_3d")
def atleast_3d(g: jit_utils.GraphContext, self: torch._C.Value) -> Value: ...
@_onnx_symbolic("prim::ConstantChunk")
def prim_constant_chunk(g: jit_utils.GraphContext, self, chunks, dim) -> list[Any]: ...
@_onnx_symbolic("aten::hstack")
def hstack(g: jit_utils.GraphContext, tensor_list: _C.Value) -> Any: ...
@_onnx_symbolic("aten::vstack")
def vstack(g: jit_utils.GraphContext, tensor_list: _C.Value): ...
