from collections.abc import Callable

from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

import torch
import torch.utils._pytree as pytree

aten = ...

def wrap_combine_fn_flat(
    *args, combine_fn, spec_init, spec_xs, num_init_leaves, num_inp_leaves
): ...
def stack_y(y: torch.Tensor, scan_length: int) -> torch.Tensor: ...
def call_operator(operator, *args) -> list[Any]: ...
def scan(
    combine_fn: Callable[
        [pytree.PyTree, pytree.PyTree], tuple[pytree.PyTree, pytree.PyTree]
    ],
    init: pytree.PyTree,
    xs: pytree.PyTree,
    *,
    dim: int = ...,
    reverse: bool = ...,
) -> tuple[pytree.PyTree, pytree.PyTree]: ...

class ScanOp(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, combine_fn, init, xs, additional_inputs) -> Any | None: ...
    def gen_schema(self, combine_fn, init, xs, additional_inputs) -> FunctionSchema: ...

scan_op = ...

def generic_scan(
    operator, init, xs, dim=..., additional_inputs=...
) -> tuple[Any, list[Any]] | list[Any | Tensor | None]: ...
def trace_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    init: list[torch.Tensor],
    xs: list[torch.Tensor],
    additional_inputs: tuple[torch.Tensor],
) -> tuple[Any | Tensor, ...]: ...
@scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def scan_op_dense(
    combine_fn, init, xs, additional_inputs
) -> tuple[Any, list[Any]] | list[Any | Tensor | None]: ...

class ScanAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        combine_fn,
        num_leaves_init,
        num_leaves_xs,
        num_additional_inputs,
        *operands,
    ) -> tuple[Any, ...]: ...
    @staticmethod
    def backward(ctx, *flat_grads) -> tuple[Any | Tensor | None, ...]: ...

@scan_op.py_autograd_impl
def scan_autograd(combine_fn, init, xs, additional_inputs) -> tuple[Any, ...]: ...
@scan_op.py_impl(ProxyTorchDispatchMode)
def scan_proxy_mode(
    mode, combine_fn, init, xs, additional_inputs
) -> tuple[Any | Tensor, ...]: ...
@scan_op.py_impl(FakeTensorMode)
def scan_fake_tensor_mode(
    mode, combine_fn, init, xs, additional_inputs
) -> tuple[Any | Tensor, ...]: ...
@scan_op.py_functionalize_impl
def scan_functionalize(ctx, combine_fn, init, xs, additional_inputs): ...
