from collections.abc import Callable
from typing import Any, NamedTuple

import enum

from torch.fx.graph import Node

class NSSingleResultValuesType(enum.StrEnum):
    WEIGHT = ...
    NODE_OUTPUT = ...
    NODE_INPUT = ...

class NSSubgraph(NamedTuple):
    start_node: Node
    end_node: Node
    base_op_node: Node

type NSSingleResultType = dict[str, Any]
type NSResultsType = dict[str, dict[str, dict[str, list[NSSingleResultType]]]]
type NSNodeTargetType = Callable | str
