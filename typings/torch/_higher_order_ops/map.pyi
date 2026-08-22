from collections.abc import Callable
from typing import Any
from typing_extensions import TypeVarTuple

from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

import torch
import torch.utils._pytree as pytree

class MapImpl(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, *args, **kwargs) -> None: ...

map_impl = ...

def map(
    f: Callable[[pytree.PyTree, tuple[pytree.PyTree, ...]], pytree.PyTree],
    xs: pytree.PyTree | torch.Tensor,
    *args: TypeVarTuple,
): ...

class MapAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, num_mapped_args, *flat_args) -> tuple[Any, ...]: ...
    @staticmethod
    def backward(ctx, *flat_grads) -> tuple[None, None, *tuple[Tensor | None, ...]]: ...

def trace_map(proxy_mode, func_overload, f, xs, pos_args) -> None: ...
@map_impl.py_impl(DispatchKey.CompositeExplicitAutograd)
def map_dense(f, xs, pos_args) -> PyTree: ...
@map_impl.py_autograd_impl
def map_autograd(f, xs, pos_args) -> None: ...
@map_impl.py_impl(ProxyTorchDispatchMode)
def map_proxy_torch_dispatch_mode(mode, f, xs, args) -> None: ...
@map_impl.py_impl(FakeTensorMode)
def map_fake_tensor_mode(mode, f, xs, args) -> PyTree: ...
@map_impl.py_functionalize_impl
def map_functionalize(ctx, f, xs, pos_args): ...
