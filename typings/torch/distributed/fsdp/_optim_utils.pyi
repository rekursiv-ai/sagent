from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple

from torch.distributed.fsdp._common_utils import _FSDPState
from torch.distributed.fsdp._flat_param import FlatParamHandle

import torch

logger = ...

@dataclass
class FSDPParamInfo:
    state: _FSDPState
    handle: FlatParamHandle
    param_indices: dict[str, int]
    param_requires_grad: list[bool]

def sorted_items(dictionary: dict[str, Any]) -> Iterator[tuple[str, Any]]: ...

@dataclass
class _ConsolidatedOptimState:
    tensor_state: dict[str, torch.Tensor] = ...
    zero_dim_tensor_state: dict[str, torch.Tensor] = ...
    non_tensor_state: dict[str, Any] = ...

class _PosDimTensorInfo(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype

class _OptimStateKey(NamedTuple):
    unflat_param_names: tuple[str, ...]
    is_fsdp_managed: bool

@dataclass
class StateInfo:
    tensors: dict[str, _PosDimTensorInfo]
    scalar_tensors: dict[str, torch.Tensor]
    non_tensors: dict[str, Any]
