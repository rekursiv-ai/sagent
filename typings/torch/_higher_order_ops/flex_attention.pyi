from collections.abc import Callable
from typing import Any

from torch import Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.fx.graph_module import GraphModule

import torch

class FlexAttentionHOP(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score_mod: Callable,
        block_mask: tuple,
        scale: float,
        kernel_options: dict[str, Any],
        score_mod_other_buffers: tuple = ...,
        mask_mod_other_buffers: tuple = ...,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

flex_attention = ...

class FlexAttentionBackwardHOP(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        logsumexp: torch.Tensor,
        grad_out: torch.Tensor,
        grad_logsumexp: torch.Tensor,
        fw_graph: Callable | GraphModule,
        joint_graph: GraphModule,
        block_mask: tuple,
        scale: float,
        kernel_options: dict[str, Any],
        score_mod_other_buffers: tuple = ...,
        mask_mod_other_buffers: tuple = ...,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]
    ]: ...

flex_attention_backward = ...

def math_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = ...,
    mask_mod_other_buffers: tuple = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
@flex_attention.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = ...,
    mask_mod_other_buffers: tuple = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def trace_flex_attention(
    proxy_mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = ...,
    mask_mod_other_buffers: tuple = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
@flex_attention.py_impl(ProxyTorchDispatchMode)
def flex_attention_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = ...,
    mask_mod_other_buffers: tuple = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
@flex_attention.py_functionalize_impl
def flex_attention_functionalize(
    ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = ...,
    mask_mod_other_buffers: tuple = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
@register_fake(flex_attention)
def flex_attention_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = ...,
    mask_mod_other_buffers: tuple = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def create_fw_bw_graph(
    score_mod: Callable,
    index_values: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    other_buffers: tuple[Tensor, ...],
) -> tuple[Callable, Callable]: ...

class FlexAttentionAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        fw_graph: Callable,
        joint_graph: Callable,
        block_mask: tuple[Any, ...],
        scale: float,
        kernel_options: dict[str, Any],
        mask_mod_other_buffers: tuple[Any, ...],
        *score_mod_other_buffers: tuple[Any, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @staticmethod
    def backward(
        ctx: Any, grad_out: Tensor, grad_logsumexp: Tensor, grad_max_scores: Tensor
    ) -> tuple[Tensor | None, ...]: ...

@flex_attention.py_impl(DispatchKey.Autograd)
def flex_attention_autograd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple[Tensor, ...] = ...,
    mask_mod_other_buffers: tuple[Tensor, ...] = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
@flex_attention_backward.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Callable,
    joint_graph: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple,
    mask_mod_other_buffers: tuple,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]
]: ...
def trace_flex_attention_backward(
    proxy_mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Callable | GraphModule,
    joint_graph: GraphModule,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = ...,
    mask_mod_other_buffers: tuple = ...,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]
]: ...
@flex_attention_backward.py_impl(ProxyTorchDispatchMode)
def flex_attention_backward_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Callable | GraphModule,
    joint_graph: GraphModule,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = ...,
    mask_mod_other_buffers: tuple = ...,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]
]: ...
@flex_attention_backward.py_functionalize_impl
def flex_attention_backward_functionalize(
    ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Callable | GraphModule,
    joint_graph: GraphModule,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = ...,
    mask_mod_other_buffers: tuple = ...,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]
]: ...
@register_fake(flex_attention_backward)
def flex_attention_backward_fake_tensor_mode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Callable | GraphModule,
    joint_graph: GraphModule,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = ...,
    mask_mod_other_buffers: tuple = ...,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]
]: ...
