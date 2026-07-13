from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import nn
from torch.ao.quantization import QConfigAny, QuantType
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node

import torch

from .custom_config import PrepareCustomConfig

__all__ = [
    "EMPTY_ARG_DICT",
    "NON_OBSERVABLE_ARG_DICT",
    "NON_QUANTIZABLE_WEIGHT_OPS",
    "NodeInfo",
    "ObservedGraphModuleAttrs",
    "all_node_args_except_first",
    "all_node_args_have_no_tensors",
    "assert_and_get_unique_device",
    "collect_producer_nodes",
    "create_getattr_from_value",
    "create_node_from_old_node_preserve_meta",
    "get_custom_module_class_keys",
    "get_linear_prepack_op_for_dtype",
    "get_new_attr_name_with_prefix",
    "get_non_observable_arg_indexes_and_types",
    "get_qconv_prepack_op",
    "get_skipped_module_name_and_classes",
    "graph_module_from_producer_nodes",
    "maybe_get_next_module",
    "node_arg_is_bias",
    "node_arg_is_weight",
    "return_arg_list",
]
NON_QUANTIZABLE_WEIGHT_OPS = ...

@dataclass
class ObservedGraphModuleAttrs:
    node_name_to_qconfig: dict[str, QConfigAny]
    node_name_to_scope: dict[str, tuple[str, type]]
    prepare_custom_config: PrepareCustomConfig
    equalization_node_name_to_qconfig: dict[str, Any]
    qconfig_mapping: QConfigMapping
    is_qat: bool
    observed_node_names: set[str]
    is_observed_standalone_module: bool = ...
    standalone_module_input_quantized_idxs: list[int] | None = ...
    standalone_module_output_quantized_idxs: list[int] | None = ...

def node_arg_is_weight(node: Node, arg: Any) -> bool: ...
def node_arg_is_bias(node: Node, arg: Any) -> bool: ...
def get_custom_module_class_keys(
    custom_module_mapping: dict[QuantType, dict[type, type]],
) -> list[Any]: ...
def get_linear_prepack_op_for_dtype(dtype) -> OpOverloadPacket[..., Any]: ...
def get_qconv_prepack_op(conv_op: Callable) -> Callable: ...
def get_new_attr_name_with_prefix(prefix: str) -> Callable: ...
def collect_producer_nodes(node: Node) -> list[Node] | None: ...
def graph_module_from_producer_nodes(
    root: GraphModule, producer_nodes: list[Node]
) -> GraphModule: ...
def assert_and_get_unique_device(module: torch.nn.Module) -> Any: ...
def create_getattr_from_value(
    module: torch.nn.Module,
    graph: Graph,
    prefix: str,
    value: Any,
    device: torch.device | None = ...,
) -> Node: ...
def all_node_args_have_no_tensors(
    node: Node, modules: dict[str, torch.nn.Module], cache: dict[Node, bool]
) -> bool: ...
def all_node_args_except_first(node: Node) -> list[int]: ...
def return_arg_list(arg_indices: list[int]) -> Callable[[Node], list[int]]: ...

NodeInfo = ...
NON_OBSERVABLE_ARG_DICT: dict[
    NodeInfo, dict[type | torch.dtype, Callable[[Node], list[int]]]
] = ...
EMPTY_ARG_DICT: dict[type | torch.dtype, Callable[[Node], list[int]]] = ...

def get_non_observable_arg_indexes_and_types(
    node: Node,
) -> dict[type | torch.dtype, Callable[[Node], list[int]]]: ...
def maybe_get_next_module(
    node: Node,
    modules: dict[str, nn.Module],
    target_module_type: type[nn.Module] | None = ...,
    target_functional_type: Any = ...,
) -> Node | None: ...
def create_node_from_old_node_preserve_meta(
    quantized_graph: Graph, create_node_args: tuple[Any, ...], old_node: Node
) -> Node: ...
def get_skipped_module_name_and_classes(
    prepare_custom_config: PrepareCustomConfig, is_standalone_module: bool
) -> tuple[list[str], list[type[Any]]]: ...
