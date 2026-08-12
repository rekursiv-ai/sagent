from collections.abc import Generator
from typing import Any
from contextlib import contextmanager as contextmanager

from torch.backends import (
    ContextProp as ContextProp,
    PropModule as PropModule,
    __allow_nonbracketed_mutation as __allow_nonbracketed_mutation,
)

__all__ = ["flags", "is_available", "set_flags"]

def is_available() -> bool: ...
def set_flags(_enabled) -> tuple[bool]: ...
@contextmanager
def flags(enabled=...) -> Generator[None, Any, None]: ...
