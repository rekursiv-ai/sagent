from collections.abc import Callable
from typing import Any

import dataclasses

from torch._higher_order_ops.triton_kernel_wrap import TraceableTritonKernelWrapper
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch.fx import GraphModule

import sympy
import torch

from .common import CodegenSymbol
from .wrapper import BufferLike, Line, PythonWrapperCodegen
from .. import ir

aten = ...
log = ...

@dataclasses.dataclass
class SymbolBuffer(CodegenSymbol):
    symbol: sympy.Symbol
    def get_name(self) -> str: ...
    def get_example(self) -> torch.Tensor | sympy.Symbol: ...

type CodegenBuffer = BufferLike | SymbolBuffer

@dataclasses.dataclass
class TritonKernel:
    tuner: CachingAutotuner
    wrapped: TraceableTritonKernelWrapper

def replace_floor_div(expr: sympy.Expr) -> sympy.Expr: ...

class WrapperFxCodegen(PythonWrapperCodegen):
    supports_caching = ...
    def compile_graph(self, gm: GraphModule) -> Callable[..., Any]: ...
    @classmethod
    def create(
        cls,
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: ir.GraphPartitionSignature | None = ...,
    ) -> WrapperFxCodegen: ...

@dataclasses.dataclass
class FxConverter:
    lines: list[Line]
    prologue: str = ...
    def __post_init__(self) -> None: ...
    def generate(self) -> torch.fx.GraphModule: ...
