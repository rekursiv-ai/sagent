from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx.passes.tools_common import NodeList

@compatibility(is_backward_compatible=False)
def topo_sort(nodes: NodeList) -> NodeList: ...
@compatibility(is_backward_compatible=False)
def validate_partition(partition: NodeList) -> bool: ...
@compatibility(is_backward_compatible=False)
def fuse_as_graphmodule(
    gm: GraphModule,
    nodes: NodeList,
    module_name: str,
    partition_lookup_table: dict[Node, int | None] | None = ...,
    *,
    always_return_tuple: bool = ...,
) -> tuple[GraphModule, tuple[Node, ...], tuple[Node, ...]]: ...
@compatibility(is_backward_compatible=False)
def insert_subgm(
    gm: GraphModule,
    sub_gm: GraphModule,
    orig_inputs: tuple[Node, ...],
    orig_outputs: tuple[Node, ...],
) -> GraphModule: ...
@compatibility(is_backward_compatible=False)
def erase_nodes(gm: GraphModule, nodes: NodeList) -> None: ...
@compatibility(is_backward_compatible=False)
def fuse_by_partitions(
    gm: GraphModule,
    partitions: list[dict[Node, int | None]],
    prefix: str = ...,
    always_return_tuple: bool = ...,
) -> GraphModule: ...
