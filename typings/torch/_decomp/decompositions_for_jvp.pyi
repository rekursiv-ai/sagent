from typing import Any
from collections.abc import Callable

from torch import Tensor

import torch

decomposition_table = ...
decomposition_table_for_jvp: dict[torch._ops.OperatorBase, Callable] = ...
register_decomposition = ...
aten = ...

def maybe_register_decomposition(op) -> Callable[..., Callable[..., Any] | Any]: ...

decomposition_table_for_jvp = ...

def register_decomposition_for_jvp(
    fn,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
@maybe_register_decomposition(aten.trace.default)
def trace(self: Tensor) -> Tensor: ...
@maybe_register_decomposition(aten.log_sigmoid_forward.default)
def log_sigmoid_forward(self: Tensor) -> tuple[Tensor, Tensor]: ...
def recompute_mean_var(
    input: Tensor, rstd: Tensor, inner_dim_indices: list[int], keepdim: bool
) -> tuple[Tensor, Tensor]: ...
@register_decomposition_for_jvp(aten.native_layer_norm_backward)
def native_layer_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: list[int],
    mean: Tensor,
    rstd: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    output_mask: list[bool],
) -> tuple[Tensor | None, Tensor | None, Tensor | None]: ...
def prod(x: list[int]) -> int: ...
@register_decomposition_for_jvp(aten.native_batch_norm_backward)
def native_batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    save_mean: Tensor | None,
    save_invstd: Tensor | None,
    train: bool,
    eps: float,
    output_mask: list[bool],
) -> tuple[Tensor, Tensor | None, Tensor | None]: ...
@register_decomposition_for_jvp(aten.batch_norm_backward)
def batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    save_mean: Tensor | None,
    save_var: Tensor | None,
    update: bool,
    eps: float,
    output_mask: list[bool],
    reserve: Tensor,
) -> tuple[Tensor, Tensor | None, Tensor | None]: ...
