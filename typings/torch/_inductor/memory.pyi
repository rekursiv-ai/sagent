from collections.abc import Callable
from typing import TypedDict

import dataclasses

from torch.utils._ordered_set import OrderedSet

from .scheduler import BaseSchedulerNode, SchedulerBuffer

torch_log = ...

@dataclasses.dataclass
class PeakMemoryResult:
    order: list[BaseSchedulerNode]
    peak_memory: int
    method: str

@dataclasses.dataclass
class MemoryPlanningInfoForBuffer:
    size_alloc: int = ...
    size_free: int = ...
    succ_nodes: OrderedSet[BaseSchedulerNode] = ...

@dataclasses.dataclass
class MemoryPlanningInfoForNode:
    index: int = ...
    size: int = ...
    pred_buffers: OrderedSet[SchedulerBuffer | FreeableInputBuffer] = ...
    pred_nodes: OrderedSet[BaseSchedulerNode] = ...
    succ_nodes: OrderedSet[BaseSchedulerNode] = ...

@dataclasses.dataclass
class FreeableInputBuffer:
    name: str
    mpi_buffer: MemoryPlanningInfoForBuffer = ...
    def get_name(self) -> str: ...
    def __hash__(self) -> int: ...

def get_freeable_input_buf(
    nodes: list[BaseSchedulerNode], graph_inputs: OrderedSet[str]
) -> dict[str, FreeableInputBuffer]: ...
def compute_size_for_scheduler_buffer(
    name_to_buf: dict[str, SchedulerBuffer],
) -> dict[str, tuple[int, int]]: ...
def assign_memory_planning_info_for_scheduler_buffers(
    nodes: list[BaseSchedulerNode], name_to_buf: dict[str, SchedulerBuffer]
) -> None: ...
def assign_memory_planning_info_for_scheduler_nodes(
    nodes: list[BaseSchedulerNode],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
) -> None: ...

@dataclasses.dataclass
class BufferInfo:
    buffer: SchedulerBuffer | FreeableInputBuffer
    size_alloc: int
    size_free: int
    start_step: int
    end_step: int

def compute_memory_timeline(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
) -> tuple[
    list[BufferInfo],
    dict[BaseSchedulerNode, int],
    dict[FreeableInputBuffer | SchedulerBuffer, BaseSchedulerNode],
]: ...
def estimate_peak_memory(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
) -> tuple[int, list[int]]: ...

@dataclasses.dataclass
class SNodeMemory:
    size_alloc: int
    size_free: int

def estimate_peak_memory_allocfree(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
) -> tuple[
    int,
    list[tuple[int, int]],
    dict[BaseSchedulerNode, SNodeMemory],
    dict[FreeableInputBuffer | SchedulerBuffer, BaseSchedulerNode],
]: ...
def topological_sort_lpmf(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    name_to_buf: dict[str, SchedulerBuffer],
    graph_outputs: OrderedSet[str],
) -> list[BaseSchedulerNode]:

    class NodeInfo(TypedDict): ...
    class BufferInfo(TypedDict): ...

def topological_sort_bfs(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:

    class NodeInfo(TypedDict): ...

    @dataclasses.dataclass
    class NodeWithPriority: ...

def topological_sort_dfs(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]: ...
def validate_graph_acyclic(nodes: list[BaseSchedulerNode]) -> None: ...
def validate_unique_buffer_names(
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
) -> None: ...
def prepare_planning_info(
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
    graph_outputs: OrderedSet[str],
) -> tuple[int, dict[str, FreeableInputBuffer]]: ...
def reorder_for_peak_memory(
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
    graph_outputs: OrderedSet[str],
    methods: list[Callable[..., list[BaseSchedulerNode]]] = ...,
) -> list[BaseSchedulerNode]: ...
