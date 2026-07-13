from typing import Any

import typing_extensions

from torch.fx import GraphModule

import torch

from .backend_config import BackendConfig
from .fx.custom_config import ConvertCustomConfig, FuseCustomConfig, PrepareCustomConfig
from .qconfig_mapping import QConfigMapping
from .utils import DEPRECATION_WARNING

def attach_preserved_attrs_to_model(
    model: GraphModule | torch.nn.Module, preserved_attrs: dict[str, Any]
) -> None: ...
def fuse_fx(
    model: torch.nn.Module,
    fuse_custom_config: FuseCustomConfig | dict[str, Any] | None = ...,
    backend_config: BackendConfig | dict[str, Any] | None = ...,
) -> GraphModule: ...
@typing_extensions.deprecated(DEPRECATION_WARNING)
def prepare_fx(
    model: torch.nn.Module,
    qconfig_mapping: QConfigMapping | dict[str, Any],
    example_inputs: tuple[Any, ...],
    prepare_custom_config: PrepareCustomConfig | dict[str, Any] | None = ...,
    _equalization_config: QConfigMapping | dict[str, Any] | None = ...,
    backend_config: BackendConfig | dict[str, Any] | None = ...,
) -> GraphModule: ...
@typing_extensions.deprecated(DEPRECATION_WARNING)
def prepare_qat_fx(
    model: torch.nn.Module,
    qconfig_mapping: QConfigMapping | dict[str, Any],
    example_inputs: tuple[Any, ...],
    prepare_custom_config: PrepareCustomConfig | dict[str, Any] | None = ...,
    backend_config: BackendConfig | dict[str, Any] | None = ...,
) -> GraphModule: ...
@typing_extensions.deprecated(DEPRECATION_WARNING)
def convert_fx(
    graph_module: GraphModule,
    convert_custom_config: ConvertCustomConfig | dict[str, Any] | None = ...,
    _remove_qconfig: bool = ...,
    qconfig_mapping: QConfigMapping | dict[str, Any] | None = ...,
    backend_config: BackendConfig | dict[str, Any] | None = ...,
    keep_original_weights: bool = ...,
) -> GraphModule: ...
def convert_to_reference_fx(
    graph_module: GraphModule,
    convert_custom_config: ConvertCustomConfig | dict[str, Any] | None = ...,
    _remove_qconfig: bool = ...,
    qconfig_mapping: QConfigMapping | dict[str, Any] | None = ...,
    backend_config: BackendConfig | dict[str, Any] | None = ...,
) -> GraphModule: ...
