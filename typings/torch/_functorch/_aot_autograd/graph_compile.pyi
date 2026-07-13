from collections.abc import Callable
from typing import Any

import dataclasses

from torch._subclasses import FakeTensor
from torch.fx.graph_module import GraphModule

import torch

from .schemas import AOTConfig, AOTGraphCapture, AOTState, FlatFn, ViewAndMutationMeta

"""
Functions in this module do most of the "work" of AOTAutograd.
An aot_dispatch_* function:
- Takes in the input flat_fn, flat_args, and some metadata
- Runs a set of pre compile wrappers (e.g. argument deduping)
- Runs the actual compiler
- Wraps the returned callable in a set of post compile wrappers
- Returns the wrapped callable and metadata.
"""
zip = ...
log = ...
aot_joint_log = ...
aot_graphs_log = ...
aten = ...
type DispatchReturn = tuple[Callable, ViewAndMutationMeta]

def aot_stage1_graph_capture(
    aot_state: AOTState, orig_flat_fn: FlatFn
) -> AOTGraphCapture: ...
def aot_stage2_export(
    aot_state: AOTState, aot_graph_capture: AOTGraphCapture
) -> DispatchReturn: ...
def sanitize_aot_config(input: AOTConfig) -> AOTConfig: ...
def aot_stage2_compile(
    aot_state: AOTState, aot_graph_capture: AOTGraphCapture
) -> DispatchReturn: ...
def aot_stage2_inference(
    aot_state: AOTState, aot_graph_capture: AOTGraphCapture
) -> DispatchReturn: ...
def collect_fw_donated_buffer_idxs(
    fw_ins: list[FakeTensor | None],
    user_fw_outs: list[FakeTensor | None],
    bw_outs: list[FakeTensor | None],
    saved_tensors: list[FakeTensor],
) -> list[int]: ...
def collect_bw_donated_buffer_idxs(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    fw_metadata: ViewAndMutationMeta,
) -> list[int]: ...

@dataclasses.dataclass
class InvokeSubgraphHopGraphs:
    partitioning_done: bool = ...
    old_num_fw_outputs: int | None = ...
    old_num_fw_inputs: int | None = ...
    new_fw_hop_gm: torch.fx.GraphModule | None = ...
    new_bw_hop_gm: torch.fx.GraphModule | None = ...
    new_num_sym_nodes: int | None = ...
    new_num_saved_nodes: int | None = ...

def prepare_for_partitioner(mod, num_primals, num_fw_outputs) -> GraphModule: ...
def run_joint_graph_passes_on_hops(
    joint_gm: torch.fx.GraphModule, joint_inputs: Any, aot_config: AOTConfig
) -> torch.fx.GraphModule: ...
def maybe_log_graph(
    gm,
    graph_name,
    aot_config,
    structured_log_prefix_fn,
    out_structured_logs: list[str] | None = ...,
) -> None: ...
def create_wrap_fn(fn, args) -> tuple[Callable[..., PyTree], Any]: ...
def prepare_hook_gm(aot_config, fn, args) -> GraphModule: ...
def maybe_inline_graph_saved_tensors_hooks(
    fw_module,
    bw_module,
    num_inner_fwd_outputs,
    inner_meta,
    aot_config,
    static_input_indices,
): ...
def aot_stage2_autograd(
    aot_state: AOTState, aot_graph_capture: AOTGraphCapture
) -> DispatchReturn: ...
