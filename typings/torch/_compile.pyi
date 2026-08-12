from collections.abc import Callable
from typing import TypeVar, overload
from typing_extensions import ParamSpec

"""
APIs related to torch.compile which lazily import torch._dynamo to avoid
circular dependencies.
"""
_T = TypeVar("_T")
_P = ParamSpec("_P")

@overload
def _disable_dynamo(
    fn: Callable[_P, _T], recursive: bool = ...
) -> Callable[_P, _T]: ...
@overload
def _disable_dynamo(
    fn: None = ..., recursive: bool = ...
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
