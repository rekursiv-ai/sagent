from collections.abc import Callable

from torch.fx import GraphModule
from torch.fx.graph import Node

from .ns_types import NSNodeTargetType, NSSubgraph

def add_loggers_to_model(
    gm: GraphModule,
    node_to_instrument_inputs_to_ref_node_name: dict[Node, tuple[str, str]],
    node_to_instrument_outputs_to_ref_node_name: dict[Node, tuple[str, str]],
    logger_cls: Callable,
    model_name: str,
) -> GraphModule: ...
def create_a_shadows_b(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    matched_subgraph_pairs: dict[str, tuple[NSSubgraph, NSSubgraph]],
    logger_cls: Callable,
    should_log_inputs: bool,
    node_type_to_io_type_map: dict[str, set[NSNodeTargetType]] | None = ...,
) -> GraphModule: ...
