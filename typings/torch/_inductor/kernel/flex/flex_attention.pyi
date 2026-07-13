from dataclasses import dataclass

import sympy
import torch

from .common import SubgraphResults
from ...ir import ComputedBuffer, TensorBox
from ...lowering import register_lowering
from ...select_algorithm import SymbolicGridFn

"""Triton Implementation of the flex_attention Kernel"""
log = ...
aten = ...
Expr = sympy.Expr

@SymbolicGridFn
def flex_attention_grid(
    batch_size, q_heads, num_queries, d_model, meta, *, cdiv
) -> tuple[Any, Any, Any]: ...
def get_float32_precision() -> Literal[ieee, tf32]: ...

flex_attention_template = ...

@register_lowering(torch.ops.higher_order.flex_attention, type_promotion_kind=None)
def flex_attention(
    query,
    key,
    value,
    subgraph,
    block_mask,
    scale,
    kernel_options,
    score_mod_other_buffers,
    mask_mod_other_buffers,
) -> (
    tuple[TensorBox | ShapeAsConstantBuffer | Any]
    | tuple[Any, Any]
    | tuple[
        TensorBox | ShapeAsConstantBuffer | Any,
        TensorBox | ShapeAsConstantBuffer,
        TensorBox | ShapeAsConstantBuffer,
    ]
): ...
@SymbolicGridFn
def flex_attention_backward_grid(
    batch_size, q_heads, num_queries, d_model, kv_heads, num_key_value, meta, *, cdiv
) -> tuple[Any, Any, Any]: ...

flex_attention_backward_template = ...

def validate_joint_graph(joint_graph: torch.fx.Graph) -> None: ...

@dataclass(frozen=True)
class JointOutputResult:
    grad_input: ComputedBuffer
    captured_grads_compute: list[ComputedBuffer]
    captured_grads: list[TensorBox | None]
    mutated_grads: list[TensorBox]

def process_joint_outputs(
    all_joint_outputs: SubgraphResults, num_placeholders: int
) -> JointOutputResult: ...
@register_lowering(
    torch.ops.higher_order.flex_attention_backward, type_promotion_kind=None
)
def flex_attention_backward(
    *args, **kwargs
) -> tuple[
    TensorBox | ShapeAsConstantBuffer,
    TensorBox | ShapeAsConstantBuffer | Any,
    TensorBox | ShapeAsConstantBuffer | Any,
    tuple[TensorBox | None, ...],
]: ...
def get_bwd_subgraph_outputs(
    subgraph_buffer: SubgraphResults,
    mask_graph_buffer: SubgraphResults,
    joint_outputs: JointOutputResult,
) -> list[ComputedBuffer | TensorBox | None]: ...
