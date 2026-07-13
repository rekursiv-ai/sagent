from typing_extensions import TypeIs

from torch import Tensor, device, dtype

class Parameter(Tensor):
    def __init__(self, data: Tensor = ..., requires_grad: bool = ...) -> None: ...

def is_lazy(param: Tensor) -> TypeIs[UninitializedParameter | UninitializedBuffer]: ...

class UninitializedParameter(Tensor):
    def __init__(self, data: Tensor = ..., requires_grad: bool = ...) -> None: ...
    def materialize(
        self,
        shape: tuple[int, ...],
        device: device | None = ...,
        dtype: dtype | None = ...,
    ) -> None: ...

class Buffer(Tensor):
    persistent: bool
    def __init__(
        self, data: Tensor = ..., requires_grad: bool = ..., persistent: bool = ...
    ) -> None: ...

class UninitializedBuffer(Tensor):
    persistent: bool
    def __init__(
        self, data: Tensor = ..., requires_grad: bool = ..., persistent: bool = ...
    ) -> None: ...
    def materialize(
        self,
        shape: tuple[int, ...],
        device: device | None = ...,
        dtype: dtype | None = ...,
    ) -> None: ...
