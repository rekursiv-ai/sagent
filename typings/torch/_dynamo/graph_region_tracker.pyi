from collections.abc import Callable
from typing import Any, TypeVar

import pickle

import torch.fx

from .symbolic_convert import InstructionTranslatorBase

"""
This module provides functionality for tracking and managing regions in computational graphs.
It supports graph optimization by identifying and grouping similar regions based on their
structure and behavior. The module implements algorithms for:

1. Tracking nodes and their relationships in the computational graph
2. Identifying identical or similar regions across the graph
3. Managing graph regions for optimization purposes
4. Supporting deduplication and other graph transformation passes

The core functionality revolves around the GraphRegionTracker class which maintains
mappings between nodes and their duplicates, enabling efficient graph analysis and
optimization operations.
"""
T = TypeVar("T")
Node = torch.fx.Node
type Region = list[Node]
type IdenticalNodes = list[Node]
type GlobalStateKey = tuple[
    bool, bool, int, bool, bool, torch.dtype, bool, bool, bool, bool
]
log = ...
graph_expansion_log = ...

def debug_log(msg: str, *args) -> None: ...

class NodeHashException(Exception): ...

class InputPickler(pickle.Pickler):
    def __init__(self) -> None: ...
    def dumps(self, obj: Any) -> bytes: ...

def get_global_state_key() -> GlobalStateKey: ...

class BackwardBfsArgIter:
    def __init__(self, origin: Node) -> None: ...
    @staticmethod
    def create(origin: Node) -> BackwardBfsArgIter: ...
    def next(self) -> Node | None: ...
    def peek(self) -> Node | None: ...
    def add_children(self, node: Node) -> None: ...

class GraphRegionTracker:
    def __init__(self) -> None: ...
    def track_node(self, tx: InstructionTranslatorBase, node: Node) -> None: ...
    def track_node_mutations(
        self,
        node: Node,
        flat_args_kwargs: list[Any],
        id_to_initial_version: dict[int, int],
    ) -> None: ...
    def add_node_mutation(self, node: Node, arg_pos: int) -> None: ...
    def get_identical_regions(self, graph: torch.fx.Graph) -> list[list[Region]]: ...

class RegionWrapper:
    def __init__(
        self, region: Region, node_to_recursive_ancestors: dict[Node, set[Node]]
    ) -> None: ...
    def next_candidate(self) -> Node | None: ...
    def will_inclusion_create_cycle(self, node: Node) -> bool: ...
    def add(self, node: Node) -> None: ...

def fully_expand_region_group(
    regions: list[Region],
    seen_nodes: set[Node],
    node_to_recursive_ancestors: dict[Node, set[Node]],
    is_identical_fn: Callable[[Node, Node], bool],
) -> None: ...
