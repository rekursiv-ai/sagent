from collections.abc import Iterable, Iterator
from typing import Any

from torch.distributed import ProcessGroup

import torch

__all__ = [
    "average_parameters",
    "average_parameters_or_parameter_groups",
    "get_params_to_average",
]

def average_parameters(
    params: Iterator[torch.nn.Parameter], process_group: ProcessGroup
) -> None: ...
def get_params_to_average(
    params: Iterable[torch.nn.Parameter] | Iterable[dict[str, torch.nn.Parameter]],
) -> list[Any]: ...
def average_parameters_or_parameter_groups(
    params: Iterable[torch.nn.Parameter] | Iterable[dict[str, torch.nn.Parameter]],
    process_group: ProcessGroup,
) -> None: ...
