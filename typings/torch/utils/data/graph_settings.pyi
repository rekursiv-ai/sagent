from typing import Any
from typing_extensions import deprecated

from torch.utils.data.graph import DataPipe, DataPipeGraph

import torch

__all__ = [
    "apply_random_seed",
    "apply_sharding",
    "apply_shuffle_seed",
    "apply_shuffle_settings",
    "get_all_graph_pipes",
]

def get_all_graph_pipes(graph: DataPipeGraph) -> list[DataPipe]: ...
def apply_sharding(
    datapipe: DataPipe, num_of_instances: int, instance_id: int, sharding_group=...
) -> DataPipe: ...
def apply_shuffle_settings(
    datapipe: DataPipe, shuffle: bool | None = ...
) -> DataPipe: ...
@deprecated(
    "`apply_shuffle_seed` is deprecated since 1.12 and will be removed in the future releases. Please use `apply_random_seed` instead.",
    category=FutureWarning,
)
def apply_shuffle_seed(datapipe: DataPipe, rng: Any) -> DataPipe: ...
def apply_random_seed(datapipe: DataPipe, rng: torch.Generator) -> DataPipe: ...
