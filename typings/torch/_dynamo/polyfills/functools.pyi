from collections.abc import Callable, Iterable
from typing import TypeVar

import functools

from ..decorators import substitute_in_graph

"""
Python polyfills for functools
"""
__all__ = ["reduce"]
_T = TypeVar("_T")
_U = TypeVar("_U")

class _INITIAL_MISSING: ...

@substitute_in_graph(functools.reduce)
def reduce(
    function: Callable[[_U, _T], _U], iterable: Iterable[_T], initial: _U = ..., /
) -> _U: ...
