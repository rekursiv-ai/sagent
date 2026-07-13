from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F")

def exposed_in(module: str) -> Callable[[F], F]: ...
