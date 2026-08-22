from collections.abc import Callable
from typing import Any

from torch import Tensor
from torch._C import DispatchKey
from torch._functorch.utils import exposed_in
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

import torch

log = ...

class CondOp(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, pred, true_fn, false_fn, operands) -> None: ...
    def gen_schema(self, pred, true_fn, false_fn, operands) -> FunctionSchema: ...

cond_op = ...

@exposed_in("torch")
def cond(
    pred: bool | float | torch.Tensor,
    true_fn: Callable,
    false_fn: Callable,
    operands: tuple | list = ...,
) -> Any: ...
def trace_cond(proxy_mode, func_overload, pred, true_fn, false_fn, operands): ...
@cond_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def cond_op_dense(pred, true_fn, false_fn, operands): ...

class CondAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, true_fn, false_fn, *operands) -> None: ...
    @staticmethod
    def backward(
        ctx, *flat_grads
    ) -> tuple[None, None, None, *tuple[Tensor | None, ...]]: ...

@cond_op.py_autograd_impl
def cond_autograd(pred, true_fn, false_fn, operands) -> None: ...
@cond_op.py_impl(ProxyTorchDispatchMode)
def inner(mode, pred, true_fn, false_fn, operands): ...
@cond_op.py_impl(FakeTensorMode)
def cond_fake_tensor_mode(mode, pred, true_fn, false_fn, operands) -> PyTree: ...
def check_tensor_meta_match(
    t1: torch.Tensor, t2: torch.Tensor, attr_names: tuple[str, ...], msg_prefix: str
) -> None: ...
@cond_op.py_functionalize_impl
def cond_func(ctx, pred, true_fn, false_fn, inputs): ...
@cond_op.py_impl(torch._C._functorch.TransformType.Vmap)
def cond_batch_rule(
    interpreter, pred, true_fn, false_fn, inputs
) -> tuple[Tensor, ...]: ...
