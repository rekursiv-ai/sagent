from typing import Any

import torch

from .descriptors import AOTInput
from .schemas import AOTConfig, FxValue, SubclassMeta, TraceFn, ViewAndMutationMeta

"""
This module dispatches the graphs to either the forward-only or joint compilation
pathways, taking into account the AOTConfig and the collected ViewAndMutationMetadata.
"""
aot_graphs_log = ...

def aot_dispatch_base_graph(
    flat_fn: TraceFn,
    flat_args: list[FxValue],
    flat_args_descs: list[AOTInput],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> tuple[
    torch.fx.GraphModule, list[FxValue], list[AOTInput], SubclassMeta | None
]: ...
def aot_dispatch_autograd_graph(
    flat_fn: TraceFn,
    flat_args: list[Any],
    flat_args_descs: list[AOTInput],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> tuple[
    torch.fx.GraphModule,
    tuple[list[Any], list[Any]],
    tuple[list[AOTInput], list[AOTInput]],
    SubclassMeta | None,
]: ...
