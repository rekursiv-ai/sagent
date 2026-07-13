from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch

doc = ...

class TracableCreateParameter(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: Any, placeholder: Any) -> torch.nn.Parameter: ...
    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[None, torch.Tensor]: ...

def tracable_create_parameter(
    tensor: torch.Tensor, placeholder: torch.nn.Parameter
) -> torch.nn.Parameter: ...
def new_parameter_placeholder(
    size: tuple[int, ...], dtype: torch.dtype, device: torch.device, requires_grad: bool
) -> torch.nn.Parameter: ...

_TLS = ...

@contextmanager
def do_not_convert_to_tracable_parameter() -> Generator[bool]: ...
def can_convert_to_tracable_parameter() -> bool: ...
