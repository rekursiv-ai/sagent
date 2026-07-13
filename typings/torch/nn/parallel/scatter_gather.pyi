from collections.abc import Sequence
from typing import Any, TypeVar, overload
from typing_extensions import deprecated

import torch

__all__ = ["gather", "scatter", "scatter_kwargs"]

@deprecated(
    "`is_namedtuple` is deprecated, please use the python checks instead",
    category=FutureWarning,
)
def is_namedtuple(obj: Any) -> bool: ...

T = TypeVar("T", dict, list, tuple)

@overload
def scatter(
    inputs: torch.Tensor, target_gpus: Sequence[int | torch.device], dim: int = ...
) -> tuple[torch.Tensor, ...]: ...
@overload
def scatter(
    inputs: T, target_gpus: Sequence[int | torch.device], dim: int = ...
) -> list[T]: ...
def scatter(
    inputs, target_gpus, dim=...
) -> (
    Any
    | list[Any]
    | list[tuple[Any, ...]]
    | list[list[Any]]
    | list[dict[Any, Any]]
    | list[Any | tuple[()] | list[Any] | dict[Any, Any]]
    | None
): ...
def scatter_kwargs(
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    target_gpus: Sequence[int | torch.device],
    dim: int = ...,
) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]: ...
def gather(outputs: Any, target_device: int | torch.device, dim: int = ...) -> Any: ...
