from torch import Tensor
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

import torch

AOTI_LOWERED_MODULE = ...

class AOTICallDelegate(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(
        self,
        lowered_module: AOTI_LOWERED_MODULE,
        original_gm: torch.fx.GraphModule,
        weight_args: list[torch.Tensor],
        input_args: list[torch.Tensor],
    ) -> list[torch.Tensor]: ...

aoti_call_delegate = ...

@aoti_call_delegate.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
def call_delegate_cpu(
    lowered_module: AOTI_LOWERED_MODULE,
    original_gm: torch.fx.GraphModule,
    weight_args: list[torch.Tensor],
    input_args: list[torch.Tensor],
) -> list[torch.Tensor]: ...
def trace_aoti_call_delegate(
    proxy_mode, func_overload, lowered_module, original_gm, weight_args, input_args
) -> list[Tensor]: ...
@aoti_call_delegate.py_impl(ProxyTorchDispatchMode)
def call_delegate_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    lowered_module: AOTI_LOWERED_MODULE,
    original_gm: torch.fx.GraphModule,
    weight_args: list[torch.Tensor],
    input_args: list[torch.Tensor],
) -> list[Tensor]: ...
@aoti_call_delegate.py_impl(FakeTensorMode)
def call_delegate_fake_tensor_mode(
    mode: FakeTensorMode,
    lowered_module: AOTI_LOWERED_MODULE,
    original_gm: torch.fx.GraphModule,
    weight_args: list[torch.Tensor],
    input_args: list[torch.Tensor],
) -> list[torch.Tensor]: ...
@aoti_call_delegate.py_functionalize_impl
def call_delegate_functionalize(
    ctx,
    lowered_module: AOTI_LOWERED_MODULE,
    original_gm: torch.fx.GraphModule,
    weight_args: list[torch.Tensor],
    input_args: list[torch.Tensor],
): ...
