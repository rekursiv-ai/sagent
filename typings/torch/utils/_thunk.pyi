from collections.abc import Callable
from typing import Generic, TypeVar

R = TypeVar("R")

class Thunk(Generic[R]):
    f: Callable[[], R] | None
    r: R | None
    __slots__ = ...
    def __init__(self, f: Callable[[], R]) -> None: ...
    def force(self) -> R: ...
