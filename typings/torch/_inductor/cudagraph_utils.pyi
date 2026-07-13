from collections.abc import (
    Callable,
    Sequence,
    Set as AbstractSet,
)
from enum import Enum
from typing import Any

import dataclasses

from torch._inductor.utils import GraphPartitionMap, InputType
from torch.utils._ordered_set import OrderedSet

import torch

perf_hint_log = ...
static_inputs_log = ...
type OutputType = list[int | torch.Tensor | None]
type ModelType = Callable[[list[InputType]], OutputType]

@dataclasses.dataclass(frozen=True)
class FunctionID:
    id: int

@dataclasses.dataclass(frozen=True)
class PlaceholderInfo:
    name: str
    stack_trace: str | None
    users: list[PlaceholderInfo]
    mutating_use_stack_trace: str | None

@dataclasses.dataclass(frozen=True)
class WrappedFunction:
    model: Callable[..., Any]
    static_input_idxs: Sequence[int]
    id: FunctionID
    constants: tuple[torch.Tensor, ...]
    placeholders: Sequence[PlaceholderInfo]
    mutated_input_idxs: Sequence[int]

def get_mutating_use_stack_trace_from_node(
    placeholder_node: torch.fx.Node,
) -> str | None: ...
def get_mutating_use_stack_trace(placeholder_info: PlaceholderInfo) -> str | None: ...
def to_placeholder_info(placeholder_node: torch.fx.Node) -> PlaceholderInfo: ...
def get_placeholder_info(graph: torch.fx.Graph) -> list[PlaceholderInfo]: ...
def format_default_skip_message(reason: str) -> str: ...
def get_mutation_stack_trace(
    placeholders: Sequence[PlaceholderInfo],
    mutation_indices: AbstractSet[int] | Sequence[int],
) -> str: ...
def check_for_mutation(
    func: WrappedFunction,
    inputs: list[InputType],
    is_cuda_graph_recorded_tensor: Callable[[torch.Tensor], bool],
) -> str | None: ...
def check_multiple_devices_or_any_cpu_nodes(
    device_node_mapping: dict[torch.device, torch.fx.Node],
) -> str | None: ...
def check_lowering_disable_cudagraph(
    device_node_mapping: dict[torch.device, torch.fx.Node],
) -> str | None: ...
def log_cudagraph_skip_and_bump_counter(msg: str) -> None: ...

@dataclasses.dataclass
class BoxedDeviceIndex:
    value: int | None
    def set(self, device_idx: int | None) -> None: ...

def check_for_mutation_ignore_cuda_graph_managed_tensor(
    gm: torch.fx.GraphModule,
    mutated_inputs: OrderedSet[str],
    mutated_input_idxs: OrderedSet[int],
    static_input_idxs: Sequence[int],
) -> str | None: ...
def get_placeholder_stack_trace(placeholder: PlaceholderInfo) -> str | None: ...

class CheckInvariantStatus(Enum):
    SUCCESS = ...
    CudagraphManagedIdxMismatch = ...
    StaticInputIdxMismatch = ...
    ExpectedDeadIndicesBeforeGraphMismatch = ...

def log_data_ptr_mismatch(
    placeholders: Sequence[PlaceholderInfo],
    inputs: list[InputType],
    recorded_data_ptr: Sequence[int | None],
    target_idxs: Sequence[int],
    mismatch: CheckInvariantStatus,
) -> str: ...
def maybe_warning_due_to_dynamic_shape(
    fn_cache: dict[tuple[int, ...], Callable[..., Any]], new_int_key: Any
) -> bool: ...

@dataclasses.dataclass(frozen=True)
class CudagraphCachedInfo:
    placeholders: Sequence[PlaceholderInfo]
    stack_traces: list[str | None]
    cudagraph_fail_reasons: list[str]

@dataclasses.dataclass(frozen=True)
class CudagraphMetadata:
    placeholders: Sequence[PlaceholderInfo]
    static_input_idxs: OrderedSet[int]
    mutated_input_idxs: OrderedSet[int]
    stack_traces: list[str | None]
    constants: dict[str, torch.Tensor]

def get_partition_cudagraph_metadata(
    partition_map: GraphPartitionMap, metadata: CudagraphMetadata
) -> CudagraphMetadata: ...
