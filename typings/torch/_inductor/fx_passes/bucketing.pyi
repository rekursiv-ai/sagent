from collections.abc import Callable
from typing import Any

import logging

from torch.utils._ordered_set import OrderedSet

import torch

logger: logging.Logger = ...

def bucket_cap_mb_by_bucket_idx_default(bucket_id: int) -> float: ...
def bucket_all_gather(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = ...,
    mode: str | None = ...,
) -> None: ...
def bucket_reduce_scatter(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = ...,
    mode: str | None = ...,
) -> None: ...
def is_all_gather_into_tensor(node: torch.fx.Node) -> bool: ...
def is_reduce_scatter_tensor(node: torch.fx.Node) -> bool: ...
def is_wait_tensor(node: torch.fx.Node) -> bool: ...
def is_wait_tensor_from_all_gather_into_tensor(node: torch.fx.Node) -> bool: ...
def collect_node_descendants(
    graph: torch.fx.Graph,
) -> dict[torch.fx.Node, OrderedSet[torch.fx.Node]]: ...
def greedy_bucket_collective_by_mb(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_node: Callable[[torch.fx.Node], bool],
    node_group_key: Callable[[torch.fx.Node], Any],
    filter_wait_node: Callable[[torch.fx.Node], bool] | None = ...,
) -> list[list[torch.fx.Node]]: ...
def bucket_all_gather_by_mb(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_wait_node: Callable[[torch.fx.Node], bool] | None = ...,
) -> list[list[torch.fx.Node]]: ...
def bucket_reduce_scatter_by_mb(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_wait_node: Callable[[torch.fx.Node], bool] | None = ...,
) -> list[list[torch.fx.Node]]: ...
def reduce_scatter_merge_fn_to_trace_custom_ops(
    rs_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    reduce_op: str,
    reduce_dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]: ...
def reduce_scatter_merge_fn_to_trace(
    rs_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    reduce_op: str,
    reduce_dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]: ...
def all_gather_merge_fn_to_trace_custom_ops(
    ag_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    dtype: torch.dtype,
    rank: int,
) -> list[torch.Tensor]: ...
def all_gather_merge_fn_to_trace(
    ag_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    dtype: torch.dtype,
    rank: int,
) -> list[torch.Tensor]: ...
def all_gather_merge_fn_to_trace_functional(
    ag_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    dtype: torch.dtype,
    rank: int,
    use_fsdp_ag_copy_in: bool = ...,
) -> list[torch.Tensor]: ...
def merge_reduce_scatter(
    gm: torch.fx.GraphModule,
    rs_buckets: list[list[torch.fx.Node]],
    mode: str | None = ...,
) -> None: ...
def merge_all_gather(
    gm: torch.fx.GraphModule,
    ag_buckets: list[list[torch.fx.Node]],
    mode: str | None = ...,
) -> None: ...
