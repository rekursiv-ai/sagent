from torch.utils._ordered_set import OrderedSet

import torch
import torch.fx

from .graph_region_tracker import Node

"""
This module implements graph deduplication functionality for TorchDynamo's optimization pipeline.
Graph deduplication identifies identical subgraphs in the computational graph and merges them
to reduce redundancy and improve performance. The process involves analyzing regions of the graph,
identifying structurally equivalent regions, and replacing them with a single shared implementation.
This optimization is particularly effective for models with repeated patterns or similar computational
structures across different parts of the network.
"""
type UsageIndex = tuple[int, int]
log = ...
last_node_to_additional_deps: dict[Node, OrderedSet[Node]] | None = ...

def apply_graph_deduplication(output_graph) -> dict[str, torch.fx.GraphModule]: ...
