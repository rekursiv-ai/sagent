from collections.abc import Sequence
from typing import Any

from torch.nn.modules import Module

import torch

__all__ = ["get_a_var", "parallel_apply"]

def get_a_var(
    obj: torch.Tensor | list[Any] | tuple[Any, ...] | dict[Any, Any],
) -> torch.Tensor | None: ...
def parallel_apply(
    modules: Sequence[Module],
    inputs: Sequence[Any],
    kwargs_tup: Sequence[dict[str, Any]] | None = ...,
    devices: Sequence[int | torch.device | None] | None = ...,
) -> list[Any]: ...
