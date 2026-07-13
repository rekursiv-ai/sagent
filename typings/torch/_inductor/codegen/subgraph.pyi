from collections.abc import Callable
from typing import Any

from torch._inductor import ir
from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.ir import Buffer, Layout

import torch

log = ...

class SubgraphChoiceCaller(ir.ChoiceCaller):
    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        description: str,
        make_fx_graph: Callable[..., Any],
    ) -> None: ...
    def benchmark(self, *args: list[Any], out: torch.Tensor) -> float: ...
    def hash_key(self) -> str: ...
    def output_node(self) -> ir.TensorBox | ir.ShapeAsConstantBuffer: ...
    def info_dict(self) -> dict[str, Any]: ...
    def autoheuristic_id(self) -> str: ...

class SubgraphTemplate(KernelTemplate):
    index_counter = ...
    def __init__(self, name: str) -> None: ...
    def generate(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        make_fx_graph: Callable[..., Any],
        description: str = ...,
        **kwargs: Any,
    ) -> SubgraphChoiceCaller: ...
