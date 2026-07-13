from torch import Tensor

from .module import Module

__all__ = [
    "AlphaDropout",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "FeatureAlphaDropout",
]

class _DropoutNd(Module):
    __constants__ = ...
    p: float
    inplace: bool
    def __init__(self, p: float = ..., inplace: bool = ...) -> None: ...
    def extra_repr(self) -> str: ...

class Dropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...

class Dropout1d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...

class Dropout2d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...

class Dropout3d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...

class AlphaDropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...

class FeatureAlphaDropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...
