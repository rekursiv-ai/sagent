from collections.abc import Callable
from typing import (
    Concatenate as _Concatenate,
    Literal,
    TypeVar as _TypeVar,
)
from typing_extensions import (
    ParamSpec as _ParamSpec,
)

from torch import _C
from torch.onnx._internal.torchscript_exporter import jit_utils

import torch._C._onnx as _C_onnx

__all__ = [
    "_apply_params",
    "_arange_cast_helper",
    "_arange_helper",
    "_argmin_argmax_helper",
    "_as_list_type",
    "_avgpool_helper",
    "_batchnorm_helper",
    "_block_list_in_opset",
    "_embedding_bag_helper",
    "_flatten_helper",
    "_generate_wrapped_number",
    "_get_const",
    "_get_dim_for_cross",
    "_get_interpolate_attributes",
    "_get_tensor_dim_size",
    "_get_tensor_rank",
    "_get_tensor_sizes",
    "_handle_reduce_dim_none",
    "_if_scalar_type_as",
    "_index_fill_reshape_helper",
    "_interpolate_get_scales",
    "_interpolate_get_scales_and_mode",
    "_interpolate_get_scales_if_available",
    "_interpolate_helper",
    "_interpolate_size_to_scales",
    "_interpolate_warning",
    "_is_bool",
    "_is_constant",
    "_is_fp",
    "_is_list",
    "_is_none",
    "_is_onnx_constant",
    "_is_packed_list",
    "_is_scalar_list",
    "_is_split_static",
    "_is_tensor",
    "_is_tensor_list",
    "_is_tuple_construct",
    "_is_value",
    "_linalg_vector_norm_helper",
    "_lt_helper",
    "_max_helper",
    "_maybe_cast_reduce_op_input",
    "_maybe_cast_to_type",
    "_maybe_get_const",
    "_maybe_get_scalar",
    "_min_helper",
    "_node_get",
    "_numel_helper",
    "_onnx_opset_unsupported",
    "_onnx_opset_unsupported_detailed",
    "_onnx_unsupported",
    "_op_with_optional_float_cast",
    "_optional_input_placeholder_tensor",
    "_overload_by_arg_count",
    "_parse_arg",
    "_reduce_op_symbolic_helper",
    "_reduce_with_dtype_helper",
    "_reducesum_helper",
    "_repeat_interleave_single_value_repeat_helper",
    "_repeat_interleave_split_helper",
    "_reshape_helper",
    "_scalar",
    "_scatter_helper",
    "_select_helper",
    "_size_helper",
    "_slice_helper",
    "_sort_helper",
    "_squeeze_helper",
    "_topk_helper",
    "_try_get_scalar_type",
    "_type_promote_from_values",
    "_unbind_helper",
    "_unimplemented",
    "_unpack_list",
    "_unpack_quantized_tensor",
    "_unpack_tuple",
    "_unsqueeze_helper",
    "_var_mean_helper",
    "args_have_same_dtype",
    "cast_pytorch_to_onnx",
    "check_training_mode",
    "dequantize_helper",
    "is_complex_value",
    "parse_args",
    "pytorch_name_to_type",
    "quantize_helper",
    "quantized_args",
    "requantize_bias_helper",
    "scalar_name_to_pytorch",
    "scalar_type_to_onnx",
    "scalar_type_to_pytorch_type",
]
_T = _TypeVar("_T")
_U = _TypeVar("_U")
_P = _ParamSpec("_P")
type _ValueDescriptor = Literal["v", "i", "is", "f", "fs", "b", "s", "t", "none"]

def parse_args(
    *arg_descriptors: _ValueDescriptor,
) -> Callable[
    [Callable[_Concatenate[_U, _P], _T]], Callable[_Concatenate[_U, _P], _T]
]: ...
def quantized_args(
    *arg_q_descriptors: bool,
    scale: float | None = ...,
    zero_point: int | None = ...,
    quantize_output: bool = ...,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
def is_complex_value(x: _C.Value) -> bool: ...
def check_training_mode(op_train_mode: int, op_name: str) -> None: ...
def dequantize_helper(
    g: jit_utils.GraphContext,
    qtensor: _C.Value,
    qdtype: _C_onnx.TensorProtoDataType | None = ...,
) -> tuple[_C.Value, _C.Value, _C.Value, _C.Value | None]: ...
def quantize_helper(
    g: jit_utils.GraphContext,
    tensor: _C.Value,
    scale: _C.Value,
    zero_point: _C.Value,
    axis: _C.Value | None = ...,
) -> _C.Value: ...
def requantize_bias_helper(
    g: jit_utils.GraphContext, bias, input_scale, weight_scale, axis=...
): ...
def args_have_same_dtype(args) -> bool: ...

cast_pytorch_to_onnx = ...
scalar_name_to_pytorch = ...
scalar_type_to_pytorch_type = ...
pytorch_name_to_type = ...
scalar_type_to_onnx = ...
_quantized_ops: set[int] = ...
