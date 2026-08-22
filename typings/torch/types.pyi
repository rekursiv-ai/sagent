from collections.abc import Sequence
from typing import IO, Any, Self

import os

from torch import (
    DispatchKey as DispatchKey,
    Size as Size,
    SymBool as SymBool,
    SymFloat as SymFloat,
    SymInt as SymInt,
    Tensor as Tensor,
    device as _device,
    dtype as _dtype,
    layout as _layout,
)
from torch.autograd.graph import GradientEdge

# The private aliases are re-exported too: `torch._C` imports `_dtype`/`_device`
# from here, and an __all__ that omits them makes every `Tensor.dtype` Unknown.
__all__ = [
    "Device",
    "FileLike",
    "Number",
    "Storage",
    "_bool",
    "_complex",
    "_device",
    "_dtype",
    "_float",
    "_int",
    "_layout",
]
type _TensorOrTensors = Tensor | Sequence[Tensor]
type _TensorOrTensorsOrGradEdge = (
    Tensor | Sequence[Tensor] | GradientEdge | Sequence[GradientEdge]
)
type _size = Size | list[int] | tuple[int, ...]
type _symsize = Size | Sequence[int | SymInt]
type _dispatchkey = str | DispatchKey
type IntLikeType = int | SymInt
type FloatLikeType = float | SymFloat
type BoolLikeType = bool | SymBool
py_sym_types = ...
type PySymType = SymInt | SymFloat | SymBool
type Number = int | float | bool
_Number = ...
type FileLike = str | os.PathLike[str] | IO[bytes]
type Device = _device | str | int | None

class Storage:
    _cdata: int
    device: _device
    dtype: _dtype
    _torch_load_uninitialized: bool
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...
    def element_size(self) -> int: ...
    def is_shared(self) -> bool: ...
    def share_memory_(self) -> Self: ...
    def nbytes(self) -> int: ...
    def cpu(self) -> Self: ...
    def data_ptr(self) -> int: ...
    def from_file(
        self, filename: str, shared: bool = ..., nbytes: int = ...
    ) -> Self: ...

_int = int
_bool = bool
_float = float
_complex = complex
