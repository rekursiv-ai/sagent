from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar
from typing_extensions import ParamSpec

from torch.utils._ordered_set import OrderedSet

import torch

from . import ir
from .graph import GraphLowering
from .virtualized import WrapperHandler

"""Utilities for lowering subgraphs used by higher order operators"""
T = TypeVar("T")
_P = ParamSpec("_P")
OpOverload = torch._ops.OpOverload
type LoweringDict = dict[OpOverload | str, Callable[..., Any]]
type TargetType = Callable[..., Any] | str

class PointwiseSubgraphLowering(torch.fx.Interpreter):
    graph_outputs: list[ir.IRNode] | None
    root_graph: GraphLowering
    _current_op: TargetType | None
    allowed_mutations: OrderedSet[OpOverload] | None
    additional_lowerings: LoweringDict | None
    buffers: list[ir.Buffer]
    mutated_buffers: OrderedSet[str]
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        root_graph_lowering: GraphLowering,
        allowed_mutations: OrderedSet[OpOverload] | None = ...,
        additional_lowerings: LoweringDict | None = ...,
    ) -> None: ...
    def mark_buffer_mutated(self, name: str) -> None: ...
    def register_buffer(self, buffer: ir.Buffer, *, set_name: bool = ...) -> str: ...
    def __getattr__(self, name: str) -> Any: ...
    def call_function(
        self, target: TargetType, args: Any, kwargs: dict[str, Any]
    ) -> Any: ...
    def output(self, target: str, args: tuple[Any], kwargs: dict[str, Any]) -> None: ...

@dataclass
class InputDescriptor:
    dtype: torch.dtype
    device: torch.device

class TracingOpsHandler(WrapperHandler):
    def __init__(self, tracer: torch.fx.Tracer, num_inputs: int) -> None: ...
    def placeholder(self, idx: int) -> torch.fx.Proxy: ...
    def output(self, *args: tuple[object]) -> None: ...

def lower_pointwise_subgraph(
    subgraph: ir.Subgraph, inputs: list[InputDescriptor]
) -> Callable[_P, Any]: ...
