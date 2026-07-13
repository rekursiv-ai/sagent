from typing import Any

import os

from torch.distributed.checkpoint.planner import LoadPlanner

def load(
    state_dict: dict[str, Any],
    *,
    checkpoint_id: str | os.PathLike[str] | None = ...,
    storage_reader: object | None = ...,
    planner: LoadPlanner | None = ...,
    process_group: object | None = ...,
    no_dist: bool = ...,
) -> None: ...
