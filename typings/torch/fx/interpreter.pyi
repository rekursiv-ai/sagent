from typing import Any

import torch

from ._compatibility import compatibility
from .graph import Graph
from .graph_module import GraphModule
from .node import Argument, Node, Target
from .proxy import Proxy

__all__ = ["Interpreter", "Transformer"]

@compatibility(is_backward_compatible=True)
class Interpreter:
    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        module: torch.nn.Module,
        garbage_collect_values: bool = ...,
        graph: Graph | None = ...,
    ) -> None: ...
    @compatibility(is_backward_compatible=True)
    def run(
        self,
        *args,
        initial_env: dict[Node, Any] | None = ...,
        enable_io_processing: bool = ...,
    ) -> Any: ...
    @compatibility(is_backward_compatible=True)
    def boxed_run(self, args_list) -> Any: ...
    @compatibility(is_backward_compatible=True)
    def run_node(self, n: Node) -> Any: ...
    @compatibility(is_backward_compatible=True)
    def placeholder(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any: ...
    @compatibility(is_backward_compatible=True)
    def get_attr(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any: ...
    @compatibility(is_backward_compatible=True)
    def call_function(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any: ...
    @compatibility(is_backward_compatible=True)
    def call_method(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any: ...
    @compatibility(is_backward_compatible=True)
    def call_module(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any: ...
    @compatibility(is_backward_compatible=True)
    def output(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any: ...
    @compatibility(is_backward_compatible=True)
    def fetch_attr(self, target: str) -> Module | Any: ...
    @compatibility(is_backward_compatible=True)
    def fetch_args_kwargs_from_env(self, n: Node) -> tuple[tuple, dict]: ...
    @compatibility(is_backward_compatible=True)
    def map_nodes_to_values(self, args: Argument, n: Node) -> Argument: ...

@compatibility(is_backward_compatible=True)
class Transformer(Interpreter):
    @compatibility(is_backward_compatible=True)
    def __init__(self, module) -> None: ...
    @compatibility(is_backward_compatible=True)
    def placeholder(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy: ...
    @compatibility(is_backward_compatible=True)
    def get_attr(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy: ...
    @compatibility(is_backward_compatible=True)
    def call_module(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any: ...
    @compatibility(is_backward_compatible=True)
    def call_function(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any: ...
    @compatibility(is_backward_compatible=True)
    def transform(self) -> GraphModule: ...
