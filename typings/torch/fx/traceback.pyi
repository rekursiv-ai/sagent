from contextlib import contextmanager
from enum import Enum
from typing import Any

from ._compatibility import compatibility
from .graph import Graph
from .node import Node

log = ...
__all__ = [
    "NodeSource",
    "NodeSourceAction",
    "format_stack",
    "get_current_meta",
    "get_graph_provenance_json",
    "has_preserved_node_meta",
    "preserve_node_meta",
    "reset_grad_fn_seq_nr",
    "set_current_meta",
    "set_grad_fn_seq_nr",
    "set_stack_trace",
]
current_meta: dict[str, Any] = ...
should_preserve_node_meta = ...

@compatibility(is_backward_compatible=False)
class NodeSourceAction(Enum):
    CREATE = ...
    REPLACE = ...

@compatibility(is_backward_compatible=False)
class NodeSource:
    class NodeInfo:
        def __init__(self, name: str, target: str, graph_id: int) -> None: ...

    pass_name: str
    action: list[NodeSourceAction]
    from_node: list[NodeSource]
    node_info: NodeInfo | None
    _dict: dict[str, Any] | None
    _action_string: str | None
    def __init__(
        self,
        node: Node | None,
        pass_name: str = ...,
        action: NodeSourceAction | list[NodeSourceAction] | None = ...,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def target(self) -> str: ...
    @property
    def graph_id(self) -> int: ...
    def print_readable(self, indent=...) -> str: ...
    def to_dict(self) -> dict: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@compatibility(is_backward_compatible=False)
@contextmanager
def preserve_node_meta(enable=...) -> Generator[None, Any, None]: ...
@compatibility(is_backward_compatible=False)
def set_stack_trace(stack: list[str]) -> None: ...
@compatibility(is_backward_compatible=False)
def set_grad_fn_seq_nr(seq_nr) -> None: ...
@compatibility(is_backward_compatible=False)
def reset_grad_fn_seq_nr() -> None: ...
@compatibility(is_backward_compatible=False)
def format_stack() -> list[str]: ...
@compatibility(is_backward_compatible=False)
def has_preserved_node_meta() -> bool: ...
@compatibility(is_backward_compatible=False)
@contextmanager
def set_current_meta(node, pass_name=...) -> Generator[None, Any, None]: ...
@compatibility(is_backward_compatible=False)
def get_current_meta() -> dict[str, Any]: ...
@compatibility(is_backward_compatible=False)
def get_graph_provenance_json(graph: Graph) -> dict[str, Any]: ...
