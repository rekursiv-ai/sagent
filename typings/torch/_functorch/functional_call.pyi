from collections.abc import Sequence
from typing import Any

from torch import Tensor, nn
from torch._functorch.utils import exposed_in

import torch

@exposed_in("torch.func")
def functional_call(
    module: torch.nn.Module,
    parameter_and_buffer_dicts: dict[str, Tensor] | Sequence[dict[str, Tensor]],
    args: Any | tuple | None = ...,
    kwargs: dict[str, Any] | None = ...,
    *,
    tie_weights: bool = ...,
    strict: bool = ...,
) -> Any: ...
@exposed_in("torch.func")
def stack_module_state(
    models: Sequence[nn.Module] | nn.ModuleList,
) -> tuple[dict[str, Any], dict[str, Any]]: ...
def construct_stacked_leaf(
    tensors: tuple[Tensor, ...] | list[Tensor], name: str
) -> Tensor: ...
