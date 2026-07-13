from collections.abc import Callable
from typing import Concatenate, TypeVar
from typing_extensions import ParamSpec

_P = ParamSpec("_P")
_T = TypeVar("_T")
_C = TypeVar("_C")
_cache_sentinel = ...

def cache_method(
    f: Callable[Concatenate[_C, _P], _T],
) -> Callable[Concatenate[_C, _P], _T]: ...
