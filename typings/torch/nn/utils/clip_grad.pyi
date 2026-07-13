from collections.abc import Callable
from typing import TypeVar
from typing_extensions import ParamSpec, deprecated

import typing

import torch

__all__: list[str] = ...
type _tensor_or_tensors = torch.Tensor | typing.Iterable[torch.Tensor]
_P = ParamSpec("_P")
_R = TypeVar("_R")

def _no_grad(func: Callable[_P, _R]) -> Callable[_P, _R]: ...
def _get_total_norm(
    tensors: _tensor_or_tensors,
    norm_type: float = ...,
    error_if_nonfinite: bool = ...,
    foreach: bool | None = ...,
) -> torch.Tensor: ...
def _clip_grads_with_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    total_norm: torch.Tensor,
    foreach: bool | None = ...,
) -> None: ...
@_no_grad
def clip_grad_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = ...,
    error_if_nonfinite: bool = ...,
    foreach: bool | None = ...,
) -> torch.Tensor: ...
@deprecated(
    "`torch.nn.utils.clip_grad_norm` is now deprecated in favor of `torch.nn.utils.clip_grad_norm_`.",
    category=FutureWarning,
)
def clip_grad_norm(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = ...,
    error_if_nonfinite: bool = ...,
    foreach: bool | None = ...,
) -> torch.Tensor: ...
@_no_grad
def clip_grad_value_(
    parameters: _tensor_or_tensors, clip_value: float, foreach: bool | None = ...
) -> None: ...
