"""The omitted-keyword sentinel.

Importable from any CLI / tool / non-tensor library without dragging in heavy
tensor dependencies.
"""

from __future__ import annotations

from typing import ClassVar, Self, override


__all__ = [
    "ABSENT",
    "Absent",
]


class Absent:
    """Sentinel for omitted keyword values. Compare with ``is ABSENT``."""

    __slots__ = ()

    _instance: ClassVar[Self | None] = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @override
    def __repr__(self) -> str:
        return "ABSENT"

    def __bool__(self) -> bool:
        return False

    @override
    def __reduce__(self) -> str:
        return "ABSENT"


ABSENT = Absent()
