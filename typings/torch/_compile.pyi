from collections.abc import Callable
from typing import TypeVar, overload
from typing_extensions import ParamSpec

"""
APIs related to torch.compile which lazily import torch._dynamo to avoid
circular dependencies.
"""
_T = TypeVar("_T")
_P = ParamSpec("_P")
_C = TypeVar("_C", bound=Callable[..., object])

# The identity overload comes FIRST and is typed on the callable itself, not
# on `Callable[_P, _T]`. A ParamSpec cannot capture an OVERLOADED function, so
# the parameterized form silently collapses every decorated overload set to
# its last signature -- which erased `Optimizer.step`'s overloads and made
# every torch optimizer fail `OptimizerProtocol` structurally while satisfying
# it at runtime. Binding the whole callable preserves the overloads verbatim.
@overload
def _disable_dynamo(fn: _C, recursive: bool = ...) -> _C: ...
@overload
def _disable_dynamo(
    fn: None = ..., recursive: bool = ...
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
