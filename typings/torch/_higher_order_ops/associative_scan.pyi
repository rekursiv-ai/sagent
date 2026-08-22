from collections.abc import Callable
from typing import Any

from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

import torch
import torch.utils._pytree as pytree

aten = ...

def wrap_combine_fn_flat(*args, combine_fn, spec, num_leaves): ...
def safe_map(f, *args) -> list[Any]: ...

class AssociativeScanOp(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, combine_fn, xs, additional_inputs) -> None: ...
    def gen_schema(self, combine_fn, xs, additional_inputs) -> FunctionSchema: ...

associative_scan_op = ...

def associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    xs: pytree.PyTree,
    dim: int,
    reverse: bool = ...,
    combine_mode: str = ...,
) -> torch.Tensor: ...
def generic_associative_scan(
    operator, leaves, dim=..., additional_inputs=...
) -> list[Any]: ...
def trace_associative_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    xs: list[torch.Tensor],
    additional_inputs: tuple[torch.Tensor],
) -> tuple[Any, ...]: ...
@associative_scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def associative_scan_op_dense(combine_fn, xs, additional_inputs) -> list[Any]: ...

class AssociativeScanAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, combine_fn, num_xs, num_additional_inputs, *operands
    ) -> tuple[Any, ...]: ...
    @staticmethod
    def backward(ctx, *gl_ys) -> tuple[Any | None, ...]: ...

@associative_scan_op.py_autograd_impl
def associative_scan_autograd(combine_fn, xs, additional_inputs) -> tuple[Any, ...]: ...
@associative_scan_op.py_impl(ProxyTorchDispatchMode)
def associative_scan_proxy_mode(
    mode, combine_fn, xs, additional_inputs
) -> tuple[Any, ...]: ...
@associative_scan_op.py_impl(FakeTensorMode)
def assoiciative_scan_fake_tensor_mode(
    mode, combine_fn, xs, additional_inputs
) -> tuple[Any, ...]: ...
@associative_scan_op.py_functionalize_impl
def associative_scan_functionalize(ctx, combine_fn, xs, additional_inputs): ...
