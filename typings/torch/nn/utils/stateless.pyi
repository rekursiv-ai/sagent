from typing import Any
from typing_extensions import deprecated

from torch import Tensor

import torch

__all__ = ["functional_call"]

@deprecated(
    "`torch.nn.utils.stateless.functional_call` is deprecated as of PyTorch 2.0 and will be removed in a future version of PyTorch. Please use `torch.func.functional_call` instead which is a drop-in replacement.",
    category=FutureWarning,
)
def functional_call(
    module: torch.nn.Module,
    parameters_and_buffers: dict[str, Tensor],
    args: Any | tuple | None = ...,
    kwargs: dict[str, Any] | None = ...,
    *,
    tie_weights: bool = ...,
    strict: bool = ...,
) -> Any: ...
