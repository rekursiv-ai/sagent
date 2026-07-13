from collections.abc import Callable
from typing import Any

from torch.ao.ns.fx.ns_types import NSResultsType
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.fx import GraphModule, Node

import torch

SHADOW_NODE_NAME_PREFIX = ...
SHADOW_WRAPPER_NODE_NAME_PREFIX = ...
BINARY_FUNCTIONS = ...

class OutputProp:
    def __init__(self, mod) -> None: ...
    def propagate(self, *args) -> None: ...

def create_submodule_from_subgraph(
    model: torch.nn.Module, first_node: Node, last_node: Node
) -> GraphModule: ...
def create_one_transformed_and_logged_copy_of_subgraph(
    mt: GraphModule,
    subgraph_idx: int,
    subgraph_candidate_idx: int,
    first_node: Node,
    last_node: Node,
    fqn: str | None,
    list_of_node_name_to_qconfig: list[dict[str, QConfigAny]],
    example_inputs: Any,
    last_added_shadow_node_list: list[Node | None],
    custom_prepare_fn: Callable | None = ...,
    custom_prepare_kwargs: dict[str, Any] | None = ...,
) -> None: ...
def create_n_transformed_and_logged_copies_of_subgraph(
    mt: GraphModule,
    subgraph_idx: int,
    match_name: str,
    nodes_in_this_subgraph: list[Any],
    qconfig_mappings: list[QConfigMapping],
    list_of_node_name_to_qconfig: list[dict[str, QConfigAny]],
    custom_prepare_fn: Callable | None = ...,
    custom_prepare_kwargs: dict[str, Any] | None = ...,
) -> None: ...
def create_add_loggers_graph(
    model: GraphModule,
    subgraphs_dedup: dict[str, list[Node]],
    qconfig_mapping: QConfigMapping,
    node_name_to_qconfig: dict[str, QConfigAny],
) -> None: ...
def extract_weight_comparison(m: GraphModule) -> NSResultsType: ...
def group_results_by_subgraph(results: NSResultsType) -> Any: ...
def create_results_comparison(results_grouped) -> Any: ...
def print_n_shadows_summary(results_comparison) -> None: ...
