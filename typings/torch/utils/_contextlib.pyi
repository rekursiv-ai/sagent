from collections.abc import Callable
from typing import Any, TypeVar

import types

type FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

def context_decorator(
    ctx, func
) -> (
    _Wrapped[..., Any, ..., Generator[Any, Any, Any]] | _Wrapped[..., Any, ..., Any]
): ...

class _DecoratorContextManager:
    def __call__(self, orig_func: F) -> F: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
    def clone(self) -> Self: ...

class _NoParamDecoratorContextManager(_DecoratorContextManager):
    def __new__(cls, orig_func=...) -> Self: ...
