from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any

import contextlib

from torch import nn
from torch.distributed.fsdp._common_utils import _FSDPState
from torch.distributed.fsdp._init_utils import ProcessGroupType
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    OptimStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictSettings,
    StateDictType,
)
from torch.distributed.tensor import DeviceMesh

import torch
import torch.distributed as dist

from .wrap import CustomPolicy, ModuleWrapPolicy

__all__ = ["FullyShardedDataParallel", "OptimStateKeyType"]
FLAT_PARAM = ...

class OptimStateKeyType(Enum):
    PARAM_NAME = ...
    PARAM_ID = ...

class FullyShardedDataParallel(nn.Module, _FSDPState):
    def __init__(
        self,
        module: nn.Module,
        process_group: ProcessGroupType = ...,
        sharding_strategy: ShardingStrategy | None = ...,
        cpu_offload: CPUOffload | None = ...,
        auto_wrap_policy: Callable | ModuleWrapPolicy | CustomPolicy | None = ...,
        backward_prefetch: BackwardPrefetch | None = ...,
        mixed_precision: MixedPrecision | None = ...,
        ignored_modules: Iterable[torch.nn.Module] | None = ...,
        param_init_fn: Callable[[nn.Module], None] | None = ...,
        device_id: int | torch.device | None = ...,
        sync_module_states: bool = ...,
        forward_prefetch: bool = ...,
        limit_all_gathers: bool = ...,
        use_orig_params: bool = ...,
        ignored_states: Iterable[torch.nn.Parameter]
        | None
        | Iterable[torch.nn.Module] = ...,
        device_mesh: DeviceMesh | None = ...,
    ) -> None: ...
    @property
    def module(self) -> nn.Module: ...
    def __getattr__(self, name: str) -> Any: ...
    def __getitem__(self, key: int) -> Any: ...
    def check_is_root(self) -> bool: ...
    @staticmethod
    def fsdp_modules(
        module: nn.Module, root_only: bool = ...
    ) -> list[FullyShardedDataParallel]: ...
    def apply(self, fn: Callable[[nn.Module], None]) -> FullyShardedDataParallel: ...
    @staticmethod
    def set_state_dict_type(
        module: nn.Module,
        state_dict_type: StateDictType,
        state_dict_config: StateDictConfig | None = ...,
        optim_state_dict_config: OptimStateDictConfig | None = ...,
    ) -> StateDictSettings: ...
    @staticmethod
    def get_state_dict_type(module: nn.Module) -> StateDictSettings: ...
    @staticmethod
    @contextlib.contextmanager
    def state_dict_type(
        module: nn.Module,
        state_dict_type: StateDictType,
        state_dict_config: StateDictConfig | None = ...,
        optim_state_dict_config: OptimStateDictConfig | None = ...,
    ) -> Generator: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    @staticmethod
    @contextlib.contextmanager
    def summon_full_params(
        module: nn.Module,
        recurse: bool = ...,
        writeback: bool = ...,
        rank0_only: bool = ...,
        offload_to_cpu: bool = ...,
        with_grads: bool = ...,
    ) -> Generator: ...
    def named_buffers(self, *args, **kwargs) -> Iterator[tuple[str, torch.Tensor]]: ...
    def named_parameters(
        self, *args, **kwargs
    ) -> Iterator[tuple[str, torch.nn.Parameter]]: ...
    @contextmanager
    def no_sync(self) -> Generator: ...
    @torch.no_grad()
    def clip_grad_norm_(
        self, max_norm: float, norm_type: float = ...
    ) -> torch.Tensor: ...
    @staticmethod
    def full_optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = ...,
        rank0_only: bool = ...,
        group: dist.ProcessGroup | None = ...,
    ) -> dict[str, Any]: ...
    @staticmethod
    def sharded_optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        group: dist.ProcessGroup | None = ...,
    ) -> dict[str, Any]: ...
    @staticmethod
    def shard_full_optim_state_dict(
        full_optim_state_dict: dict[str, Any],
        model: torch.nn.Module,
        optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = ...,
        optim: torch.optim.Optimizer | None = ...,
    ) -> dict[str, Any]: ...
    @staticmethod
    def flatten_sharded_optim_state_dict(
        sharded_optim_state_dict: dict[str, Any],
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
    ) -> dict[str, Any]: ...
    @staticmethod
    def scatter_full_optim_state_dict(
        full_optim_state_dict: dict[str, Any] | None,
        model: torch.nn.Module,
        optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = ...,
        optim: torch.optim.Optimizer | None = ...,
        group: Any | None = ...,
    ) -> dict[str, Any]: ...
    @staticmethod
    def rekey_optim_state_dict(
        optim_state_dict: dict[str, Any],
        optim_state_key_type: OptimStateKeyType,
        model: torch.nn.Module,
        optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = ...,
        optim: torch.optim.Optimizer | None = ...,
    ) -> dict[str, Any]: ...
    @staticmethod
    def optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_state_dict: dict[str, Any] | None = ...,
        group: dist.ProcessGroup | None = ...,
    ) -> dict[str, Any]: ...
    @staticmethod
    def optim_state_dict_to_load(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_state_dict: dict[str, Any],
        is_named_optimizer: bool = ...,
        load_directly: bool = ...,
        group: dist.ProcessGroup | None = ...,
    ) -> dict[str, Any]: ...
    def register_comm_hook(self, state: object, hook: callable) -> None: ...
