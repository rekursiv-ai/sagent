from collections.abc import Callable, Iterable
from typing import Any, TypeVar, overload
from typing_extensions import TypeVarTuple

import operator

from ..decorators import substitute_in_graph

"""
Python polyfills for operator
"""
__all__ = ["attrgetter", "countOf", "itemgetter", "methodcaller"]
_T = TypeVar("_T")
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_Ts = TypeVarTuple("_Ts")
_U = TypeVar("_U")
_U1 = TypeVar("_U1")
_U2 = TypeVar("_U2")
_Us = TypeVarTuple("_Us")

@overload
def attrgetter(attr: str, /) -> Callable[[Any], _U]: ...
@overload
def attrgetter(
    attr1: str, attr2: str, /, *attrs: str
) -> Callable[[Any], tuple[_U1, _U2, *_Us]]: ...
@substitute_in_graph(operator.attrgetter, is_embedded_type=True)
def attrgetter(*attrs: str) -> Callable[[Any], Any | tuple[Any, ...]]: ...
@overload
def itemgetter(item: _T, /) -> Callable[[Any], _U]: ...
@overload
def itemgetter(
    item1: _T1, item2: _T2, /, *items: *_Ts
) -> Callable[[Any], tuple[_U1, _U2, *_Us]]: ...
@substitute_in_graph(operator.itemgetter, is_embedded_type=True)
def itemgetter(*items: Any) -> Callable[[Any], Any | tuple[Any, ...]]: ...
@substitute_in_graph(operator.methodcaller, is_embedded_type=True)
def methodcaller(name: str, /, *args: Any, **kwargs: Any) -> Callable[[Any], Any]: ...
@substitute_in_graph(operator.countOf, can_constant_fold_through=True)
def countOf(a: Iterable[_T], b: _T, /) -> int: ...
