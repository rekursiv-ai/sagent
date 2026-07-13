from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from torch import nn
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec

import torch

from ._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
from ._fsdp_common import FSDPMeshInfo

lib = ...

@torch.library.impl(lib, "copy_", "Meta")
@torch.library.impl(lib, "copy_", "CUDA")
@torch.library.impl(lib, "copy_", "XPU")
@torch.library.impl(lib, "copy_", "HPU")
@torch.library.impl(lib, "copy_", "CPU")
@torch.library.impl(lib, "copy_", "MTIA")
def copy_(tensor, data) -> None: ...
@torch.library.impl(lib, "copy_", "Functionalize")
def copy__functionalize(tensor, data) -> None: ...

class ShardedState(Enum):
    SHARDED = ...
    SHARDED_POST_FORWARD = ...
    UNSHARDED = ...

@dataclass
class ParamModuleInfo:
    module: nn.Module
    param_name: str
    shared_modules: list[nn.Module] = ...
    shared_param_names: list[str] = ...

@dataclass
class ExtensionsData:
    all_gather_metadata: Any | None = ...
    all_gather_input_sizes: Sequence[torch.Size] = ...
    def clear(self) -> None: ...

class FSDPParam:
    orig_dtype: torch.dtype
    param_dtype: torch.dtype | None
    reduce_dtype: torch.dtype | None
    _orig_size: torch.Size
    sharded_size: torch.Size
    contiguous_sharded_stride: tuple[int, ...]
    padded_sharded_param_size: torch.Size
    sharded_post_forward_size: torch.Size
    contiguous_sharded_post_forward_stride: tuple[int, ...]
    _sharded_param_data: torch.Tensor
    sharded_param: nn.Parameter
    _sharded_post_forward_param_data: torch.Tensor | None
    _sharded_post_forward_param: nn.Parameter | None
    _unsharded_param: nn.Parameter
    unsharded_accumulated_grad: torch.Tensor | None
    _sharding_spec: DTensorSpec
    _tp_spec: DTensorSpec
    all_gather_outputs: list[torch.Tensor]
    _extensions_data: ExtensionsData
    _unsharded_inner_tensors: list[torch.Tensor]
    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: FSDPMeshInfo | None,
        device: torch.device,
        shard_placement_fn: Callable[[nn.Parameter], Shard | None] | None,
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
    ) -> None: ...
    def init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy) -> None: ...
    def init_all_gather_outputs(
        self,
        all_gather_input_numels: list[int],
        all_gather_input_dtypes: list[torch.dtype],
        world_size: int,
        device: torch.device,
        force_recreate: bool = ...,
    ) -> None: ...
    def init_unsharded_param(self) -> None: ...
    def to_sharded(self) -> None: ...
    def to_sharded_post_forward(self) -> None: ...
    def to_unsharded(self) -> None: ...
    def to_sharded_dtensor(self, tensor: torch.Tensor) -> DTensor: ...
    def to_sharded_post_forward_dtensor(self, tensor: torch.Tensor) -> DTensor: ...
    def to_accumulated_grad_if_needed(self) -> None: ...
    def accumulate_unsharded_grad_if_needed(self) -> None: ...
    def alloc_all_gather_outputs(self) -> None: ...
    def free_unsharded_param(self) -> None: ...
    @property
    def all_gather_inputs(self) -> list[torch.Tensor]: ...
    @property
    def unsharded_param(self) -> nn.Parameter: ...
    @property
    def unsharded_grad_data(self) -> torch.Tensor: ...
    @property
    def unsharded_accumulated_grad_data(self) -> torch.Tensor: ...
    @property
    def shard_mesh(self) -> DeviceMesh: ...
    @property
    def shard_mesh_from_root(self) -> DeviceMesh: ...
    def reset_sharded_param(self) -> None: ...

def alloc_storage(tensor: torch.Tensor) -> None: ...
def free_storage(tensor: torch.Tensor) -> None: ...
def unsafe_setattr_param(
    module: nn.Module, param_name: str, param: nn.Parameter
) -> None: ...
def set_requires_grad_if_needed(
    src_tensor: torch.Tensor, dst_tensor: torch.Tensor
) -> None: ...
