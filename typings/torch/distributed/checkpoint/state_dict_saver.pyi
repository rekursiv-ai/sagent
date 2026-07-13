from typing import Any

import os

from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.planner import SavePlanner

def save(
    state_dict: dict[str, Any],
    *,
    checkpoint_id: str | os.PathLike[str] | None = ...,
    storage_writer: object | None = ...,
    planner: SavePlanner | None = ...,
    process_group: object | None = ...,
    no_dist: bool = ...,
    use_collectives: bool = ...,
) -> Metadata: ...
