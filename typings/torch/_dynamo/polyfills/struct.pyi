from collections.abc import Buffer
from typing import Any

import struct

from ..decorators import substitute_in_graph

"""
Python polyfills for struct
"""
__all__ = ["pack", "unpack"]

@substitute_in_graph(struct.pack, can_constant_fold_through=True)
def pack(fmt: bytes | str, /, *v: Any) -> bytes: ...
@substitute_in_graph(struct.unpack, can_constant_fold_through=True)
def unpack(format: bytes | str, buffer: Buffer, /) -> tuple[Any, ...]: ...
