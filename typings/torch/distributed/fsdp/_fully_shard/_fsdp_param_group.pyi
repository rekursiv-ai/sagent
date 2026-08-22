from collections.abc import Callable, Generator
from typing import Any, NamedTuple

import contextlib

from torch import Tensor, nn
from torch.distributed.tensor import Shard
from torch.utils.hooks import RemovableHandle

import torch

from ._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
from ._fsdp_collectives import AllGatherResult
from ._fsdp_common import FSDPMeshInfo, TrainingState

logger = ...
type _ModuleToHandleDict = dict[nn.Module, RemovableHandle]

class FSDPCommContext:
    def lazy_init(self, device: torch.device) -> None: ...
    def get_all_gather_streams(
        self, async_op: bool, training_state: TrainingState
    ) -> tuple[torch.Stream, torch.Stream]: ...

class AllGatherState(NamedTuple):
    all_gather_result: AllGatherResult
    event: torch.Event | None

class ReduceScatterState(NamedTuple):
    reduce_scatter_input: torch.Tensor
    event: torch.Event | None

class AllReduceState(NamedTuple):
    all_reduce_input: torch.Tensor
    event: torch.Event | None

class FSDPParamGroup:
    _orig_dtype: torch.dtype | None
    _reduce_dtype: torch.dtype | None
    def __init__(
        self,
        params: list[nn.Parameter],
        modules: tuple[nn.Module, ...],
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: FSDPMeshInfo | None,
        device: torch.device,
        shard_placement_fn: Callable[[nn.Parameter], Shard | None] | None,
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
    ) -> None: ...
    def lazy_init(self) -> None: ...
    def set_allocate_memory_from_process_group(self, enable: bool) -> None: ...
    def unshard(self, async_op: bool = ...) -> None: ...
    def wait_for_unshard(self) -> None: ...
    def reshard(self) -> None: ...
    def pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]: ...
    def post_forward(self, module: nn.Module, input: Any, output: Any) -> Any: ...
    def pre_backward(self, default_prefetch: bool, *unused: Any) -> None: ...
    def post_backward(self, *unused: Any) -> None: ...
    def finalize_backward(self) -> None: ...
    @property
    def is_sharded(self) -> bool: ...
    @property
    def is_sharded_post_forward(self) -> bool: ...
    @property
    def is_unsharded(self) -> bool: ...
    @contextlib.contextmanager
    def use_training_state(
        self, training_state: TrainingState
    ) -> Generator[None, Any]: ...

class RegisterPostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, param_group: FSDPParamGroup, *inputs: torch.Tensor
    ) -> tuple[Tensor, ...]: ...
    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> tuple[None, *tuple[Tensor, ...]]: ...
