from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import logging

from torch import fx
from torch.utils._ordered_set import OrderedSet

import torch

aten = ...
logger: logging.Logger = ...

def move_block_after(block: list[fx.Node], target_node: fx.Node) -> None: ...
def move_block_before(block: list[fx.Node], target_node: fx.Node) -> None: ...
def call_function(
    graph: fx.Graph,
    target: str | Callable[..., Any],
    args: tuple[fx.node.Argument, ...] | None = ...,
    kwargs: dict[str, fx.node.Argument] | None = ...,
) -> fx.Node: ...

@dataclass(unsafe_hash=True)
class CommBlock:
    shape: torch.Size | list[torch.Size]
    node_list: list[fx.Node]
    inputs: list[fx.Node]
    wait_nodes: list[fx.Node]
    comm_node: fx.Node
    outputs: OrderedSet[fx.Node]

def get_comm_block(comm_node: fx.Node) -> CommBlock | None: ...
def get_all_comm_blocks(
    graph: fx.Graph,
    comm_ops: tuple[torch._ops.OpOverload, ...],
    comm_filter: Callable[..., bool] | None = ...,
) -> list[CommBlock]: ...
def fuse_ddp_with_coalesced_op(graph: fx.Graph, bucket_size_mb: int) -> None: ...
def fuse_ddp_with_concat_op(graph: fx.Graph, bucket_size_mb: int) -> None: ...
def schedule_comm_wait(graph: fx.Graph) -> None: ...
def fuse_ddp_communication(
    graph: fx.Graph, passes: list[Callable[..., None] | str], bucket_size_mb: int
) -> None: ...
