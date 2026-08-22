from collections.abc import Callable
from typing import Any

from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

import torch

class WhileLoopOp(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        carried_inputs: tuple[torch.Tensor | int | float | bool],
        additional_inputs: tuple[torch.Tensor | torch.SymInt | int, ...],
        /,
    ) -> None: ...
    def gen_schema(
        self, cond_fn, body_fn, carried_inputs, additional_inputs
    ) -> FunctionSchema: ...

while_loop_op = ...

def while_loop(cond_fn, body_fn, carried_inputs) -> None: ...
@while_loop_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def while_loop_dense(
    cond_fn, body_fn, carried_inputs, additional_inputs, stack_output=...
) -> tuple[Tensor | Any, ...] | tuple[Tensor, ...] | tuple[Any, ...]: ...
@while_loop_op.py_autograd_impl
def while_loop_autograd(cond_fn, body_fn, operands, additional_inputs) -> None: ...
@while_loop_op.py_impl(ProxyTorchDispatchMode)
def while_loop_tracing(
    mode, cond_fn, body_fn, carried_inputs, additional_inputs, stack_output=...
): ...
@while_loop_op.py_impl(FakeTensorMode)
def while_loop_fake_tensor_mode(
    mode, cond_fn, body_fn, carried_inputs, additional_inputs, stack_output=...
) -> PyTree: ...
@while_loop_op.py_functionalize_impl
def while_loop_func(
    ctx, cond_fn, body_fn, carried_inputs, additional_inputs, stack_output=...
): ...

class WhileLoopStackOutputOp(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        carried_inputs: tuple[torch.Tensor | int | float | bool],
        additional_inputs: tuple[torch.Tensor | torch.SymInt | int, ...],
        /,
    ) -> None: ...

class WhileLoopAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cond_fn,
        body_fn,
        num_carried_inputs,
        num_additional_inputs,
        *carries_and_inputs,
    ) -> tuple[Any, ...]: ...
    @staticmethod
    def backward(
        ctx, *grads
    ) -> tuple[None, None, None, None, *tuple[Tensor | None, ...]]: ...

while_loop_stack_output_op = ...
