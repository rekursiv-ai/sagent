from torch import Tensor
from typing import Any

def pin_memory(
    data, device=...
) -> (
    Tensor
    | str
    | bytes
    | Any
    | dict[
        Any, Tensor | str | bytes | Any | dict[Any, Any] | tuple[Any, ...] | list[Any]
    ]
    | tuple[Any, ...]
    | list[Any]
): ...
