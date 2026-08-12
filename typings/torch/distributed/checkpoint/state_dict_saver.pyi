from concurrent.futures import Future
from enum import Enum
from typing import Any

import os

from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.planner import SavePlanner

class AsyncCheckpointerType(Enum):
    THREAD = ...
    PROCESS = ...

class AsyncSaveResponse:
    staging_completion: Future[None]
    upload_completion: Future[None]

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
def async_save(
    state_dict: dict[str, Any],
    *,
    checkpoint_id: str | os.PathLike[str] | None = ...,
    storage_writer: object | None = ...,
    planner: SavePlanner | None = ...,
    process_group: object | None = ...,
    async_checkpointer_type: AsyncCheckpointerType = ...,
    async_stager: object | None = ...,
    no_dist: bool = ...,
    use_collectives: bool = ...,
) -> Future[None] | AsyncSaveResponse: ...
