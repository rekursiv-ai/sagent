from typing import Literal
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import os

from torch.distributed.checkpoint.stateful import StatefulT

import torch

__all__ = [
    "BytesStorageMetadata",
    "ChunkStorageMetadata",
    "Metadata",
    "MetadataIndex",
    "StorageMeta",
    "TensorProperties",
    "TensorStorageMetadata",
]

@dataclass
class ChunkStorageMetadata:
    offsets: torch.Size
    sizes: torch.Size

class _MEM_FORMAT_ENCODING(Enum):
    TORCH_CONTIGUOUS_FORMAT = ...
    TORCH_CHANNELS_LAST = ...
    TORCH_PRESERVE_FORMAT = ...

@dataclass
class TensorProperties:
    dtype: torch.dtype = ...
    layout: torch.layout = ...
    requires_grad: bool = ...
    memory_format: torch.memory_format = ...
    pin_memory: bool = ...
    def __getstate__(
        self,
    ) -> tuple[
        dtype,
        layout,
        bool,
        Literal[
            _MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT,
            _MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST,
            _MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT,
        ],
        bool,
    ]: ...
    def __setstate__(self, state) -> None: ...
    @staticmethod
    def create_from_tensor(tensor: torch.Tensor) -> TensorProperties: ...

@dataclass
class TensorStorageMetadata:
    properties: TensorProperties
    size: torch.Size
    chunks: list[ChunkStorageMetadata]

@dataclass
class BytesStorageMetadata: ...

type STORAGE_TYPES = TensorStorageMetadata | BytesStorageMetadata
type STATE_DICT_TYPE = dict[str, StatefulT | Any]

@dataclass
class StorageMeta:
    checkpoint_id: str | os.PathLike | None = ...
    save_id: str | None = ...
    load_id: str | None = ...
    modules: list[str] = ...

@dataclass
class Metadata:
    state_dict_metadata: dict[str, STORAGE_TYPES]
    planner_data: Any = ...
    storage_data: Any = ...
    storage_meta: StorageMeta | None = ...
    version: str | None = ...

@dataclass(frozen=True)
class MetadataIndex:
    fqn: str
    offset: torch.Size | None = ...
    index: int | None = ...
    def __init__(
        self, fqn: str, offset: Sequence[int] | None = ..., index: int | None = ...
    ) -> None: ...
