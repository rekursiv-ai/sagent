from collections.abc import Sequence

from torch import Tensor
from torch.nn.common_types import _size_2_t, _size_4_t, _size_6_t

from .module import Module

__all__ = [
    "CircularPad1d",
    "CircularPad2d",
    "CircularPad3d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "ZeroPad1d",
    "ZeroPad2d",
    "ZeroPad3d",
]

class _CircularPadNd(Module):
    __constants__ = ...
    padding: Sequence[int]
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class CircularPad1d(_CircularPadNd):
    padding: tuple[int, int]
    def __init__(self, padding: _size_2_t) -> None: ...

class CircularPad2d(_CircularPadNd):
    padding: tuple[int, int, int, int]
    def __init__(self, padding: _size_4_t) -> None: ...

class CircularPad3d(_CircularPadNd):
    padding: tuple[int, int, int, int, int, int]
    def __init__(self, padding: _size_6_t) -> None: ...

class _ConstantPadNd(Module):
    __constants__ = ...
    value: float
    padding: Sequence[int]
    def __init__(self, value: float) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ConstantPad1d(_ConstantPadNd):
    padding: tuple[int, int]
    def __init__(self, padding: _size_2_t, value: float) -> None: ...

class ConstantPad2d(_ConstantPadNd):
    __constants__ = ...
    padding: tuple[int, int, int, int]
    def __init__(self, padding: _size_4_t, value: float) -> None: ...

class ConstantPad3d(_ConstantPadNd):
    padding: tuple[int, int, int, int, int, int]
    def __init__(self, padding: _size_6_t, value: float) -> None: ...

class _ReflectionPadNd(Module):
    __constants__ = ...
    padding: Sequence[int]
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ReflectionPad1d(_ReflectionPadNd):
    padding: tuple[int, int]
    def __init__(self, padding: _size_2_t) -> None: ...

class ReflectionPad2d(_ReflectionPadNd):
    padding: tuple[int, int, int, int]
    def __init__(self, padding: _size_4_t) -> None: ...

class ReflectionPad3d(_ReflectionPadNd):
    padding: tuple[int, int, int, int, int, int]
    def __init__(self, padding: _size_6_t) -> None: ...

class _ReplicationPadNd(Module):
    __constants__ = ...
    padding: Sequence[int]
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class ReplicationPad1d(_ReplicationPadNd):
    padding: tuple[int, int]
    def __init__(self, padding: _size_2_t) -> None: ...

class ReplicationPad2d(_ReplicationPadNd):
    padding: tuple[int, int, int, int]
    def __init__(self, padding: _size_4_t) -> None: ...

class ReplicationPad3d(_ReplicationPadNd):
    padding: tuple[int, int, int, int, int, int]
    def __init__(self, padding: _size_6_t) -> None: ...

class ZeroPad1d(ConstantPad1d):
    padding: tuple[int, int]
    def __init__(self, padding: _size_2_t) -> None: ...
    def extra_repr(self) -> str: ...

class ZeroPad2d(ConstantPad2d):
    padding: tuple[int, int, int, int]
    def __init__(self, padding: _size_4_t) -> None: ...
    def extra_repr(self) -> str: ...

class ZeroPad3d(ConstantPad3d):
    padding: tuple[int, int, int, int, int, int]
    def __init__(self, padding: _size_6_t) -> None: ...
    def extra_repr(self) -> str: ...
