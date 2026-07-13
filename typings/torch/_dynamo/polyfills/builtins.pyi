from collections.abc import Iterable
from typing import TypeVar

import builtins

from ..decorators import substitute_in_graph

"""
Python polyfills for builtins
"""
__all__ = ["all", "any", "enumerate", "sum"]
_T = TypeVar("_T")

@substitute_in_graph(builtins.all, can_constant_fold_through=True)
def all(iterable: Iterable[object], /) -> bool: ...
@substitute_in_graph(builtins.any, can_constant_fold_through=True)
def any(iterable: Iterable[object], /) -> bool: ...
@substitute_in_graph(builtins.enumerate, is_embedded_type=True)
def enumerate(iterable: Iterable[_T], start: int = ...) -> Iterable[tuple[int, _T]]: ...
@substitute_in_graph(builtins.sum, can_constant_fold_through=True)
def sum(iterable: Iterable[_T], /, start: _T = ...) -> _T: ...

class _CallableIterator:
    def __init__(self, fn, sentinel) -> None: ...
    def __iter__(self) -> Self: ...
    def __next__(self): ...

class _SENTINEL_MISSING: ...

def iter_(
    fn_or_iterable, sentinel=..., /
) -> Generator[Any, Any, None] | _CallableIterator: ...
