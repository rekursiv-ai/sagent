from typing import Any

import dataclasses

from torch import _C

import torch

"""Utilities for manipulating the torch.Graph object and the torchscript."""
_ATTR_PATTERN = ...
_SKIP_NODE_ATTRIBUTES = ...

@dataclasses.dataclass
class GraphContext:
    graph: _C.Graph
    block: _C.Block
    opset: int
    original_node: _C.Node
    params_dict: dict[str, _C.IValue]
    env: dict[_C.Value, _C.Value]
    values_in_env: set[_C.Value]
    new_nodes: list[_C.Node] = ...
    def __getattr__(self, name: str) -> Any: ...
    def op(
        self,
        opname: str,
        *raw_args: torch.Tensor | _C.Value,
        outputs: int = ...,
        **kwargs,
    ): ...
    def aten_op(self, operator: str, *args, overload_name: str = ..., **kwargs): ...

    at = ...
    def onnxscript_op(
        self, onnx_fn, *raw_args: torch.Tensor | _C.Value, outputs: int = ..., **kwargs
    ): ...

def add_op_with_blocks(
    graph_context: GraphContext,
    opname: str,
    *inputs: _C.Value,
    outputs: int = ...,
    n_blocks: int = ...,
    **attributes,
) -> tuple[Any, tuple[GraphContext, ...], _C.Node]: ...
def get_device_from_value(value: _C.Value) -> torch.device | None: ...
def parse_node_kind(kind: str) -> tuple[str, str]: ...
def is_aten(domain: str) -> bool: ...
def is_prim(domain: str) -> bool: ...
def is_onnx(domain: str) -> bool: ...
