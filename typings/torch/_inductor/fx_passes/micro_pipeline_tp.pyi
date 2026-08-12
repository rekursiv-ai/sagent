from typing import Any
from dataclasses import dataclass

from torch.utils._ordered_set import OrderedSet

import torch

from ..pattern_matcher import Match

log = ...
aten = ...
patterns = ...

@dataclass
class _AllGatherMatch:
    match: Match
    shard_node: torch.fx.Node
    ag_node: torch.fx.Node
    res_node: torch.fx.Node
    gather_dim: int
    group_name: str
    def replace_with(self, new_node: torch.fx.Node) -> None: ...
    def erase(self) -> None: ...

def find_all_gather_patterns(graph: torch.fx.Graph) -> list[Any]: ...

@dataclass
class _ReduceScatterMatch:
    match: Match
    input_node: torch.fx.Node
    reduce_scatter_node: torch.fx.Node
    wait_tensor_node: torch.fx.Node
    reduce_op: str
    scatter_dim: int
    group_name: str
    def replace_with(self, new_node: torch.fx.Node) -> None: ...
    def erase(self) -> None: ...

def find_reduce_scatter_patterns(graph: torch.fx.Graph) -> list[Any]: ...

@dataclass
class _Matmul:
    nodes: list[torch.fx.Node]
    arg_ancestor_nodes: OrderedSet[torch.fx.Node] = ...
    A_node: torch.fx.Node
    B_node: torch.fx.Node
    pre_mm_reshape: torch.fx.Node | None
    post_mm_reshape: torch.fx.Node | None
    def __post_init__(self) -> None: ...
    def replace_with(self, new_node: torch.fx.Node) -> None: ...
    def erase(self) -> None: ...
    @classmethod
    def from_match(cls, match: list[torch.fx.Node]) -> _Matmul: ...

@dataclass
class _ScaledMatmul(_Matmul):
    A_scale_node: torch.fx.Node
    B_scale_node: torch.fx.Node
    bias_node: torch.fx.Node | None
    result_scale_node: torch.fx.Node | None
    out_dtype: torch.dtype | None
    use_fast_accum: bool
    pre_mm_reshape: torch.fx.Node | None
    post_mm_reshape: torch.fx.Node | None
    def __post_init__(self) -> None: ...
    @classmethod
    def from_match(cls, match: list[torch.fx.Node]) -> _ScaledMatmul: ...

def fuse_all_gather_matmul(all_gather: _AllGatherMatch) -> None: ...
def fuse_matmul_reduce_scatter(reduce_scatter: _ReduceScatterMatch) -> None: ...
def micro_pipeline_tp_pass(graph: torch.fx.Graph) -> None: ...
