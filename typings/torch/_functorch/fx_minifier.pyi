from torch import Tensor
from typing import Any, Literal
from collections.abc import Callable
from dataclasses import dataclass

from torch import fx

import torch

is_tuple = ...

@dataclass
class LoadTensorMeta:
    size: list[int]
    stride: list[int]
    dtype: torch.dtype
    device: torch.device

class ConcreteProp(torch.fx.Interpreter):
    def __init__(self, mod, *, writer=..., skip_offload=...) -> None: ...
    def run_node(self, n) -> Tensor | Any: ...
    def propagate(self, *args) -> Any: ...

def is_load_tensor_node(node) -> bool: ...
def create_minified_hlo_graph(minified_fx_graph, inputs) -> None: ...
def dump_state(fx_g, inps) -> None: ...
def is_power_of_two(n) -> Literal[False]: ...

@dataclass
class ReproState:
    graph: fx.Graph
    inps: list[torch.Tensor]
    def __post_init__(self) -> None: ...

def minifier(
    fail_f: fx.GraphModule,
    inps,
    module_fails,
    dump_state: Callable = ...,
    *,
    save_dir=...,
    offload_to_disk=...,
    skip_offload=...,
    skip_sanity=...,
    max_granularity=...,
) -> tuple[GraphModule, Any | list[Tensor]]: ...
