from collections.abc import Callable
from typing import Any

from torch import nn
from torch.ao.quantization.utils import Pattern

import torch

from .backend_config import BackendConfig, DTypeConfig

__all__ = [
    "entry_to_pretty_str",
    "get_fused_module_classes",
    "get_fuser_method_mapping",
    "get_fusion_pattern_to_extra_inputs_getter",
    "get_fusion_pattern_to_root_node_getter",
    "get_module_to_qat_module",
    "get_pattern_to_dtype_configs",
    "get_pattern_to_input_type_to_index",
    "get_qat_module_classes",
    "get_root_module_to_quantized_reference_module",
    "pattern_to_human_readable",
    "remove_boolean_dispatch_from_name",
]

def get_pattern_to_dtype_configs(
    backend_config: BackendConfig,
) -> dict[Pattern, list[DTypeConfig]]: ...
def get_qat_module_classes(backend_config: BackendConfig) -> tuple[type, ...]: ...
def get_fused_module_classes(backend_config: BackendConfig) -> tuple[type, ...]: ...
def get_pattern_to_input_type_to_index(
    backend_config: BackendConfig,
) -> dict[Pattern, dict[str, int]]: ...
def get_root_module_to_quantized_reference_module(
    backend_config: BackendConfig,
) -> dict[type[torch.nn.Module], type[torch.nn.Module]]: ...
def get_fuser_method_mapping(
    backend_config: BackendConfig,
) -> dict[Pattern, nn.Sequential | Callable]: ...
def get_module_to_qat_module(
    backend_config: BackendConfig,
) -> dict[Pattern, type[torch.nn.Module]]: ...
def get_fusion_pattern_to_root_node_getter(
    backend_config: BackendConfig,
) -> dict[Pattern, Callable]: ...
def get_fusion_pattern_to_extra_inputs_getter(
    backend_config: BackendConfig,
) -> dict[Pattern, Callable]: ...
def remove_boolean_dispatch_from_name(p) -> Any: ...
def pattern_to_human_readable(p) -> Any: ...
def entry_to_pretty_str(entry) -> str: ...
