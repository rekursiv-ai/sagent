from typing import TypeVar

"""Miscellaneous utilities to aid with typing."""
T = TypeVar("T")

def not_none(obj: T | None) -> T: ...
