from typing import NamedTuple

import types

from torch import Tensor

import torch

from .grad_mode import _DecoratorContextManager

__all__ = [
    "UnpackedDualTensor",
    "dual_level",
    "enter_dual_level",
    "exit_dual_level",
    "make_dual",
    "unpack_dual",
]
_current_level = ...

def enter_dual_level() -> int: ...
def exit_dual_level(*, level=...) -> None: ...
def make_dual(tensor, tangent, *, level=...) -> Tensor: ...

class UnpackedDualTensor(NamedTuple):
    primal: torch.Tensor
    tangent: torch.Tensor | None

def unpack_dual(tensor, *, level=...) -> UnpackedDualTensor: ...

class dual_level(_DecoratorContextManager):
    def __enter__(self) -> int: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

_is_fwd_grad_enabled = ...

class _set_fwd_grad_enabled(_DecoratorContextManager):
    def __init__(self, mode: bool) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
