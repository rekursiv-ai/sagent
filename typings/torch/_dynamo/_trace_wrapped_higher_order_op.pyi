from typing import Any

from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses import FakeTensorMode
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.overrides import TorchFunctionMode

import torch

"""trace_wrapped(*args, fn) is equivalent to fn(*args), but with a twist:
if you make_fx trace through this call, we will not actually trace into fn; instead,
we will directly insert it as a call_function to fn in the graph.
(Unlike make_fx, Dynamo WILL inline into fn.)
You can think of this as a one off allow_in_graph equivalent for proxy tensor tracing.

Because proxy tensor tracing does not actually run the function, there are
requirements on the behavior of fn. We are still figuring it out, but here is the current state:

1) fn SHOULD only take a single argument, which must be a tensor
2) fn MUST return a new tensor with the same metadata as the original tensor
   (e.g., zeros_like(input) is a permissible implementation of fn).
   This is verified via an extra assert that is inserted into the traced graph.
3) fn MAY have side effects, but it MAY NOT perform metadata mutation on other tensors
   participating in proxy tensor tracing (it MAY mutate other tensors, it MAY mutate Python state)
These requirements stem from the requirement that we need to continue performing proxy tensor tracing,
which assumes accurate fake tensor metadata, without actually running fn.
In the future, we may allow for a "meta" function associated with fn to allow for more interesting input-output patterns.

Note that tensors / Python state are allowed to be mutated.
This is relaxed constraint is not always sound, but it is sound for backward tracing with fake
tensors as it takes place in AOTAutograd, as the backward pass is guaranteed not to depend on concrete
tensor values (via fake tensor) or Python state (because the autograd engine doesn't depend on Python).

The intended use case for this function is to allow AOTAutograd to defer complex
backward hooks to compiled autograd. AOTAutograd performs a make_fx trace which preserves
the function call as is in the graph, and only when we Dynamo through the backward graph in
compiled autograd do we inline into the function.
"""
Tensor = torch.Tensor
__all__ = ["trace_wrapped"]

@torch.library.custom_op("flex_lib::zeros_and_scatter", mutates_args=())
def zeros_and_scatter(
    shape: list[int], indices: list[Tensor], vals: Tensor
) -> Tensor: ...
@zeros_and_scatter.register_fake
def _(shape: list[int], indices: list[Tensor], vals: Tensor) -> Tensor: ...
@zeros_and_scatter.register_vmap
def _(info, indims, shape, indices, value) -> tuple[Any, None]: ...

class ModIndex(torch.autograd.Function):
    generate_vmap_rule = ...
    @staticmethod
    def forward(x: Tensor, indices: list[Tensor]) -> Tensor: ...
    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None: ...
    @staticmethod
    def backward(ctx, gradOut) -> tuple[Any, None]: ...
    @classmethod
    @torch._export.wrappers.allow_in_pre_dispatch_graph
    def apply(cls, *args, **kwargs) -> Any | None: ...

mod_index = ...

class TransformGetItemToIndex(TorchFunctionMode):
    def __torch_function__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = ...,
        kwargs: dict[str, object] | None = ...,
    ) -> object: ...

def trace_wrapped(*args: Any, **kwargs: Any) -> Any: ...

class TraceWrapped(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

_trace_wrapped_op = ...

@_trace_wrapped_op.py_impl(ProxyTorchDispatchMode)
def inner_trace(
    mode: ProxyTorchDispatchMode,
    *args: Any,
    bw_state: BackwardState | None = ...,
    **kwargs: Any,
) -> Any: ...
@_trace_wrapped_op.py_impl(FakeTensorMode)
def inner_fake(*args: Any, **kwargs: Any) -> None: ...
def autograd_function_backward_rewritten(original_backward: Any) -> Any: ...
