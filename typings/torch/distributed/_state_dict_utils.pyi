from collections.abc import MutableMapping
from typing import Any, NamedTuple

import torch

class CompanionMismatch(Exception): ...

class _TensorInfo(NamedTuple):
    size: torch.Size
    dtype: torch.dtype

type PATH_ITEM = str | int
type OBJ_PATH = tuple[PATH_ITEM, ...]
type FLATTEN_MAPPING = dict[str, OBJ_PATH]
type STATE_DICT_TYPE = dict[str, Any]
type CONTAINER_TYPE = MutableMapping[PATH_ITEM, Any]
