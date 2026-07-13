from collections.abc import Callable
from typing import Any

from torch.fx import GraphModule
from torch.fx.graph import Node

from .ns_types import NSNodeTargetType

toq = ...

def get_type_a_related_to_b(
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]],
) -> set[tuple[NSNodeTargetType, NSNodeTargetType]]: ...

type NSFusionElType = Callable | str | tuple[str, Any]
type NSFusionType = (
    tuple[NSFusionElType, NSFusionElType]
    | tuple[NSFusionElType, NSFusionElType, NSFusionElType, NSFusionElType]
)

def get_reversed_fusions() -> list[tuple[NSFusionType, int]]: ...
def end_node_matches_reversed_fusion(
    end_node: Node,
    reversed_fusion: NSFusionType,
    gm: GraphModule,
    seen_nodes: set[Node],
) -> bool: ...
