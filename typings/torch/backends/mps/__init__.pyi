from functools import lru_cache as _lru_cache
from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
    Optional as Optional,
)

from torch.library import Library as _Library

__all__ = [
    "get_core_count",
    "get_name",
    "is_available",
    "is_built",
    "is_macos13_or_newer",
    "is_macos_or_newer",
]

def is_built() -> bool: ...
@_lru_cache
def is_available() -> bool: ...
@_lru_cache
def is_macos_or_newer(major: int, minor: int) -> bool: ...
@_lru_cache
def is_macos13_or_newer(minor: int = ...) -> bool: ...
@_lru_cache
def get_name() -> str: ...
@_lru_cache
def get_core_count() -> int: ...

_lib: _Library | None = ...
