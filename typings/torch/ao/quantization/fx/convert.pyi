from typing import Any

from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quant_type import QuantType
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node

import torch

from .custom_config import ConvertCustomConfig

__all__ = [
    "convert",
    "convert_custom_module",
    "convert_standalone_module",
    "convert_weighted_module",
]
SUPPORTED_QDTYPES = ...
_QSCHEME_TO_CHOOSE_QPARAMS_OP = ...

def convert_standalone_module(
    node: Node,
    modules: dict[str, torch.nn.Module],
    model: torch.fx.GraphModule,
    is_reference: bool,
    backend_config: BackendConfig | None,
) -> None: ...
def convert_weighted_module(
    node: Node,
    modules: dict[str, torch.nn.Module],
    observed_node_names: set[str],
    node_name_to_qconfig: dict[str, QConfigAny],
    backend_config: BackendConfig,
    is_decomposed: bool = ...,
    is_reference: bool = ...,
    model_device: torch.device | None = ...,
) -> None: ...
def convert_custom_module(
    node: Node,
    graph: Graph,
    modules: dict[str, torch.nn.Module],
    custom_module_class_mapping: dict[QuantType, dict[type, type]],
    statically_quantized_custom_module_nodes: set[Node],
) -> None: ...
def convert(
    model: GraphModule,
    is_reference: bool = ...,
    convert_custom_config: ConvertCustomConfig | dict[str, Any] | None = ...,
    is_standalone_module: bool = ...,
    _remove_qconfig_flag: bool = ...,
    qconfig_mapping: QConfigMapping | dict[str, Any] | None = ...,
    backend_config: BackendConfig | dict[str, Any] | None = ...,
    is_decomposed: bool = ...,
    keep_original_weights: bool = ...,
) -> GraphModule: ...
