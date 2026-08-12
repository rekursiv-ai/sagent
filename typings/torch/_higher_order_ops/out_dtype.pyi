from torch import Tensor
from typing import Literal
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

import torch

ALLOWABLE_OPS = ...

class OutDtypeOperator(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, op, output_dtype, *args) -> None: ...

out_dtype = ...

def trace_out_dtype(proxy_mode, func_overload, op, output_dtype, *args) -> object: ...
@out_dtype.py_impl(DispatchKey.CompositeExplicitAutograd)
def out_dtype_dense(
    op: torch._ops.OpOverload, output_dtype: torch.dtype, *args
) -> Tensor: ...
def is_int_mm(op, output_dtype, args) -> Literal[False]: ...
def out_dtype_fallback(op, output_dtype, *args): ...
@out_dtype.py_impl(ProxyTorchDispatchMode)
def out_dtype_proxy(
    mode: ProxyTorchDispatchMode,
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args,
) -> object: ...
@out_dtype.py_impl(FakeTensorMode)
def out_dtype_fake_tensor_mode(
    mode: FakeTensorMode, op: torch._ops.OpOverload, output_dtype: torch.dtype, *args
) -> Tensor: ...
@out_dtype.py_functionalize_impl
def out_dtype_func(ctx, op, output_dtype, *args): ...
