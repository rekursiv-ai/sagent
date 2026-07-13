from dataclasses import dataclass
from enum import Enum
from typing import Any

import abc
import io

from torch.distributed.checkpoint.metadata import (
    STATE_DICT_TYPE,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
)

import torch

__all__ = [
    "BytesIOWriteData",
    "LoadItemType",
    "LoadPlan",
    "LoadPlanner",
    "ReadItem",
    "SavePlan",
    "SavePlanner",
    "TensorWriteData",
    "WriteItem",
    "WriteItemType",
]

class WriteItemType(Enum):
    TENSOR = ...
    SHARD = ...
    BYTE_IO = ...

class LoadItemType(Enum):
    TENSOR = ...
    BYTE_IO = ...

@dataclass(frozen=True)
class BytesIOWriteData:
    nbytes: int

@dataclass(frozen=True)
class TensorWriteData:
    chunk: ChunkStorageMetadata
    properties: TensorProperties
    size: torch.Size

@dataclass(frozen=True)
class WriteItem:
    index: MetadataIndex
    type: WriteItemType
    bytes_io_data: BytesIOWriteData | None = ...
    tensor_data: TensorWriteData | None = ...
    def tensor_storage_size(self) -> int | None: ...

@dataclass(frozen=True)
class ReadItem:
    type: LoadItemType
    dest_index: MetadataIndex
    dest_offsets: torch.Size
    storage_index: MetadataIndex
    storage_offsets: torch.Size
    lengths: torch.Size

@dataclass(frozen=True)
class SavePlan:
    items: list[WriteItem]
    storage_data: Any = ...
    planner_data: Any = ...
    usable: bool = ...

@dataclass
class LoadPlan:
    items: list[ReadItem]
    storage_data: Any = ...
    planner_data: Any = ...

class SavePlanner(abc.ABC):
    _cached_save_plan: dict[str, SavePlan] = ...
    _cached_final_save_plan: dict[str, SavePlan] = ...
    _cached_all_plans: dict[str, list[SavePlan]] = ...
    _cached_global_plan: dict[str, list[SavePlan]] = ...
    _cached_metadata: dict[str, Metadata] = ...
    @abc.abstractmethod
    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        storage_meta: StorageMeta | None = ...,
        is_coordinator: bool = ...,
    ) -> None: ...
    @abc.abstractmethod
    def create_local_plan(self) -> SavePlan: ...
    @abc.abstractmethod
    def create_global_plan(
        self, all_plans: list[SavePlan]
    ) -> tuple[list[SavePlan], Metadata]: ...
    @abc.abstractmethod
    def finish_plan(self, new_plan: SavePlan) -> SavePlan: ...
    @abc.abstractmethod
    def resolve_data(self, write_item: WriteItem) -> torch.Tensor | io.BytesIO: ...

class LoadPlanner:
    @abc.abstractmethod
    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = ...,
        is_coordinator: bool = ...,
    ) -> None: ...
    @abc.abstractmethod
    def create_local_plan(self) -> LoadPlan: ...
    @abc.abstractmethod
    def create_global_plan(self, global_plan: list[LoadPlan]) -> list[LoadPlan]: ...
    @abc.abstractmethod
    def finish_plan(self, central_plan: LoadPlan) -> LoadPlan: ...
    @abc.abstractmethod
    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None: ...
    def resolve_bytes(self, read_item: ReadItem) -> io.BytesIO: ...
    @abc.abstractmethod
    def resolve_tensor(self, read_item: ReadItem) -> torch.Tensor: ...
    @abc.abstractmethod
    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None: ...
