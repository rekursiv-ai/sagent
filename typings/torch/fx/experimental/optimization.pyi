from collections.abc import Callable, Iterable
from typing import Any

from torch import fx, nn

import torch

__all__ = [
    "MklSubgraph",
    "UnionFind",
    "extract_subgraph",
    "fuse",
    "gen_mkl_autotuner",
    "matches_module_pattern",
    "modules_to_mkldnn",
    "optimize_for_inference",
    "remove_dropout",
    "replace_node_module",
    "reset_modules",
    "use_mkl_length",
]

def matches_module_pattern(
    pattern: Iterable[type], node: fx.Node, modules: dict[str, Any]
) -> bool: ...
def replace_node_module(
    node: fx.Node, modules: dict[str, Any], new_module: torch.nn.Module
) -> None: ...
def fuse(model: torch.nn.Module, inplace=..., no_trace=...) -> torch.nn.Module: ...
def remove_dropout(model: nn.Module) -> nn.Module: ...
def extract_subgraph(
    orig_module: nn.Module,
    nodes: list[fx.Node],
    inputs: list[fx.Node],
    outputs: list[fx.Node],
) -> GraphModule: ...

mkldnn_supported = ...
mkldnn_supported_unknown = ...
mkldnn_map = ...

def modules_to_mkldnn(
    nodes: list[fx.Node], modules: dict[str, nn.Module]
) -> dict[Module, Module]: ...
def reset_modules(
    nodes: list[fx.Node],
    modules: dict[str, nn.Module],
    old_modules: dict[nn.Module, nn.Module],
) -> None: ...

class MklSubgraph:
    def __init__(self, fx_graph: fx.Graph) -> None: ...

def gen_mkl_autotuner(example_inputs, iters=..., warmup=...) -> Callable[..., bool]: ...
def use_mkl_length(graph: MklSubgraph) -> bool: ...

class UnionFind:
    def __init__(self, n) -> None: ...
    def make_set(self, v: int) -> None: ...
    def find(self, v: int) -> int: ...
    def join(self, a: int, b: int) -> int | None: ...

def optimize_for_inference(
    model: torch.nn.Module,
    pass_config: dict[str, Any] | None = ...,
    tracer: type[fx.Tracer] = ...,
) -> torch.nn.Module: ...
