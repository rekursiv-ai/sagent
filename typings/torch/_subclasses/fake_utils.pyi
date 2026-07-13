from collections.abc import Callable
from typing import Any

from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._python_dispatch import TorchDispatchMode

import torch

aten = ...

def outputs_alias_inputs(outputs, inputs) -> bool: ...
def outputs_are_inputs(outputs, inputs) -> bool: ...
def output_alias_each_other(outputs) -> bool: ...
def is_sdpa_error(func, idx, e) -> bool: ...
def try_convert_fake_to_real(
    ten_list: list[FakeTensor | Any],
) -> list[FakeTensor | torch.Tensor | Any]: ...

class CrossRefFakeMode(TorchDispatchMode):
    def __init__(
        self,
        ignore_op_fn: Callable[[OpOverload], bool] | None = ...,
        *,
        check_strides=...,
        check_aliasing=...,
        only_check_ops_with_meta=...,
    ) -> None: ...
    def __torch_dispatch__(self, func, types, args=..., kwargs=...): ...
