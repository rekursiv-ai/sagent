from typing import Any

from torch import Tensor

import torch
import torch.utils._pytree as pytree

from .descriptors import AOTInput
from .schemas import GraphSignature, ViewAndMutationMeta

"""
This module is one of the analysis modules - it takes as input a function or graph
and some preexisting properties, and returns some data that is useful for deciding
how to further proceed with compilation or construct runtime wrappers.

In particular, the following analyses are provided:
1. Refine the view and mutation metadata collected previously - removing duplicate
   inputs or mapping views to their bases.
2. We also analyze the function signature for export graphs.
"""
zip = ...

def remove_dupe_metadata(
    m: ViewAndMutationMeta, keep_arg_mask: list[bool], add_dupe_map: list[int]
) -> ViewAndMutationMeta: ...
def create_synthetic_base_metadata(
    m: ViewAndMutationMeta,
    synthetic_base_info: list[int | tuple[int, torch.Tensor]],
    outer_args: list[Any],
    inner_args: list[Any],
    inner_args_desc: list[AOTInput],
) -> tuple[ViewAndMutationMeta, list[int]]: ...
def compute_overlapping_inputs(
    aot_config, fwd_inputs, aliased_input_indices
) -> set[Any]: ...
def create_graph_signature(
    fx_g: torch.fx.GraphModule,
    fw_metadata: ViewAndMutationMeta,
    in_spec: pytree.TreeSpec,
    out_spec: pytree.TreeSpec,
    *,
    user_args_flat: list[Tensor],
    params_and_buffers_flat: list[Tensor],
    param_names: list[str],
    buffer_names: list[str],
    trace_joint: bool,
    num_user_fw_outs: int | None,
    loss_index: int | None,
) -> GraphSignature: ...
