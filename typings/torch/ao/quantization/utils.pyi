from typing import Literal
from collections.abc import Callable
from typing import Any

from torch.fx import Node

import torch

"""
Utils shared by different modes of quantization (eager/graph)
"""
type NodePattern = tuple[Node, Node] | tuple[Node, tuple[Node, Node]] | Any
type QuantizerCls = Any
type Pattern = (
    Callable
    | tuple[Callable, Callable]
    | tuple[Callable, tuple[Callable, Callable]]
    | Any
)

class MatchAllNode: ...

module_type_list = ...
func_list = ...
method_list = ...

def check_node(node, modules) -> tuple[Any | bool, Any | bool, Any | bool]: ...
def get_combined_dict(default_dict, additional_dict): ...
def is_per_tensor(qscheme): ...
def is_per_channel(qscheme) -> bool: ...
def getattr_from_fqn(obj: Any, fqn: str) -> Any: ...
def to_underlying_dtype(qdtype) -> dtype: ...
def get_qparam_dict(observer_or_fake_quant) -> dict[str, Any | None]: ...
def get_swapped_custom_module_class(
    custom_module, custom_module_class_mapping, qconfig
): ...
def activation_dtype(qconfig): ...
def weight_dtype(qconfig): ...
def activation_is_statically_quantized(qconfig) -> bool: ...
def activation_is_dynamically_quantized(qconfig) -> bool: ...
def activation_is_int8_quantized(qconfig) -> bool: ...
def activation_is_int32_quantized(qconfig) -> bool: ...
def weight_is_quantized(qconfig) -> bool: ...
def weight_is_statically_quantized(qconfig) -> bool: ...
def op_is_int8_dynamically_quantized(qconfig) -> bool: ...
def get_qconfig_dtypes(qconfig) -> tuple[Any, Any, Any | bool]: ...
def get_quant_type(
    qconfig,
) -> Literal[QuantType.DYNAMIC, QuantType.STATIC, QuantType.WEIGHT_ONLY]: ...
def check_min_max_valid(min_val: torch.Tensor, max_val: torch.Tensor) -> bool: ...
def calculate_qmin_qmax(
    quant_min: int,
    quant_max: int,
    has_customized_qrange: bool,
    dtype: torch.dtype,
    reduce_range: bool,
) -> tuple[int, int]: ...
def has_no_children_ignoring_parametrizations(module) -> bool: ...
def validate_qmin_qmax(quant_min: int, quant_max: int) -> None: ...
def determine_qparams(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    eps: torch.Tensor,
    has_customized_qrange: bool,
    qscheme: torch.qscheme = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def get_fqn_to_example_inputs(
    model: torch.nn.Module, example_inputs: tuple[Any, ...]
) -> dict[str, tuple[Any, ...]]: ...

DEPRECATION_WARNING = ...
__all__ = [
    "DEPRECATION_WARNING",
    "MatchAllNode",
    "NodePattern",
    "Pattern",
    "activation_dtype",
    "activation_is_dynamically_quantized",
    "activation_is_int8_quantized",
    "activation_is_int32_quantized",
    "activation_is_statically_quantized",
    "calculate_qmin_qmax",
    "check_min_max_valid",
    "check_node",
    "determine_qparams",
    "get_combined_dict",
    "get_fqn_to_example_inputs",
    "get_qconfig_dtypes",
    "get_qparam_dict",
    "get_quant_type",
    "get_swapped_custom_module_class",
    "getattr_from_fqn",
    "has_no_children_ignoring_parametrizations",
    "is_per_channel",
    "is_per_tensor",
    "op_is_int8_dynamically_quantized",
    "to_underlying_dtype",
    "validate_qmin_qmax",
    "weight_dtype",
    "weight_is_quantized",
    "weight_is_statically_quantized",
]
