from typing import Any, Literal
import torch

from ..ir import ShapeAsConstantBuffer, Subgraph, TensorBox
from ..pattern_matcher import Arg, CallFunction, Match, register_graph_pattern
from ..select_algorithm import SymbolicGridFn

B2B_GEMM_PASS = ...

@SymbolicGridFn
def b2b_gemm_grid(M, P, meta, *, cdiv) -> tuple[Any, Literal[1], Literal[1]]: ...

b2b_gemm_left_template = ...
b2b_gemm_right_template = ...

def load_ratio_left(
    M: int, N: int, O: int, P: int, m: int, n: int, o: int, p: int
) -> float: ...
def load_ratio_right(
    M: int, N: int, O: int, P: int, m: int, n: int, o: int, p: int
) -> float: ...

b2b_gemm_configs = ...

def is_b2b_gemm_good_on(
    is_left_assoc: bool,
    A_node: torch.fx.Node,
    B_node: torch.fx.Node,
    C_node: torch.fx.Node,
) -> bool: ...
def unoptimized_b2b_gemm(
    is_left_assoc: bool,
    subgraph: Subgraph,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor: ...

unoptimized_choice = ...

def build_subgraph_buffer(args: list[TensorBox], subgraph: Subgraph) -> PyTree: ...
def create_placeholder(
    name: str, dtype: torch.dtype, device: torch.device
) -> TensorBox | ShapeAsConstantBuffer: ...
def tuned_b2b_gemm(
    is_left_assoc: bool,
    subgraph: Subgraph,
    A: torch._inductor.ir.TensorBox,
    B: torch._inductor.ir.TensorBox,
    C: torch._inductor.ir.TensorBox,
    *,
    layout=...,
) -> torch._inductor.ir.TensorBox: ...
@register_graph_pattern(
    CallFunction(torch.ops.aten.mm, Arg(), Arg()), pass_dict=B2B_GEMM_PASS
)
def b2b_gemm_handler(
    match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node
) -> None: ...
