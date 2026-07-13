from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar, overload

import itertools

from ..decorators import substitute_in_graph

"""
Python polyfills for itertools
"""
__all__ = [
    "accumulate",
    "chain",
    "chain_from_iterable",
    "compress",
    "cycle",
    "dropwhile",
    "filterfalse",
    "islice",
    "tee",
    "zip_longest",
]
_T = TypeVar("_T")
_U = TypeVar("_U")
type _Predicate[_T] = Callable[[_T], object]
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")

@substitute_in_graph(itertools.chain, is_embedded_type=True)
def chain(*iterables: Iterable[_T]) -> Iterator[_T]: ...
@substitute_in_graph(itertools.accumulate, is_embedded_type=True)
def accumulate(
    iterable: Iterable[_T],
    func: Callable[[_T, _T], _T] | None = ...,
    *,
    initial: _T | None = ...,
) -> Iterator[_T]: ...
@substitute_in_graph(itertools.chain.from_iterable)
def chain_from_iterable(iterable: Iterable[Iterable[_T]], /) -> Iterator[_T]: ...
@substitute_in_graph(itertools.compress, is_embedded_type=True)
def compress(data: Iterable[_T], selectors: Iterable[_U], /) -> Iterator[_T]: ...
@substitute_in_graph(itertools.cycle, is_embedded_type=True)
def cycle(iterable: Iterable[_T]) -> Iterator[_T]: ...
@substitute_in_graph(itertools.dropwhile, is_embedded_type=True)
def dropwhile(predicate: _Predicate[_T], iterable: Iterable[_T], /) -> Iterator[_T]: ...
@substitute_in_graph(itertools.filterfalse, is_embedded_type=True)
def filterfalse(
    function: _Predicate[_T], iterable: Iterable[_T], /
) -> Iterator[_T]: ...
@substitute_in_graph(itertools.islice, is_embedded_type=True)
def islice(iterable: Iterable[_T], /, *args: int | None) -> Iterator[_T]: ...
@substitute_in_graph(itertools.pairwise, is_embedded_type=True)
def pairwise(iterable: Iterable[_T], /) -> Iterator[tuple[_T, _T]]: ...
@substitute_in_graph(itertools.tee)
def tee(iterable: Iterable[_T], n: int = ..., /) -> tuple[Iterator[_T], ...]: ...
@overload
def zip_longest(
    iter1: Iterable[_T1], /, *, fillvalue: _U = ...
) -> Iterator[tuple[_T1]]: ...
@overload
def zip_longest(
    iter1: Iterable[_T1], iter2: Iterable[_T2], /
) -> Iterator[tuple[_T1 | None, _T2 | None]]: ...
@overload
def zip_longest(
    iter1: Iterable[_T1], iter2: Iterable[_T2], /, *, fillvalue: _U = ...
) -> Iterator[tuple[_T1 | _U, _T2 | _U]]: ...
@overload
def zip_longest(
    iter1: Iterable[_T],
    iter2: Iterable[_T],
    iter3: Iterable[_T],
    /,
    *iterables: Iterable[_T],
) -> Iterator[tuple[_T | None, ...]]: ...
@overload
def zip_longest(
    iter1: Iterable[_T],
    iter2: Iterable[_T],
    iter3: Iterable[_T],
    /,
    *iterables: Iterable[_T],
    fillvalue: _U = ...,
) -> Iterator[tuple[_T | _U, ...]]: ...
@substitute_in_graph(itertools.zip_longest, is_embedded_type=True)
def zip_longest(
    *iterables: Iterable[_T], fillvalue: _U = ...
) -> Iterator[tuple[_T | _U, ...]]: ...
