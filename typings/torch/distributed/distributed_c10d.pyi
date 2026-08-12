from typing import Self
from collections.abc import Callable
from datetime import timedelta
from typing import Any
from typing_extensions import deprecated

import pickle

from torch import Tensor
from torch._C._distributed_c10d import ProcessGroup, ReduceOp, Store, Work

import torch

from .c10d_logger import _exception_logger, _time_logger

"""Distributed Collective Communication (c10d)."""
__all__ = [
    "AllToAllOptions",
    "AllreduceCoalescedOptions",
    "AllreduceOptions",
    "Backend",
    "BackendConfig",
    "BarrierOptions",
    "BroadcastOptions",
    "DebugLevel",
    "GatherOptions",
    "GroupMember",
    "P2POp",
    "PrefixStore",
    "ProcessGroup",
    "ReduceOp",
    "ReduceOptions",
    "ReduceScatterOptions",
    "ScatterOptions",
    "Store",
    "Work",
    "all_gather",
    "all_gather_coalesced",
    "all_gather_into_tensor",
    "all_gather_object",
    "all_reduce",
    "all_reduce_coalesced",
    "all_to_all",
    "all_to_all_single",
    "barrier",
    "batch_isend_irecv",
    "broadcast",
    "broadcast_object_list",
    "default_pg_timeout",
    "destroy_process_group",
    "gather",
    "gather_object",
    "get_backend",
    "get_backend_config",
    "get_debug_level",
    "get_default_backend_for_device",
    "get_global_rank",
    "get_group_rank",
    "get_node_local_rank",
    "get_pg_count",
    "get_process_group_ranks",
    "get_rank",
    "get_world_size",
    "group",
    "init_process_group",
    "irecv",
    "is_backend_available",
    "is_gloo_available",
    "is_initialized",
    "is_mpi_available",
    "is_nccl_available",
    "is_torchelastic_launched",
    "is_ucc_available",
    "is_xccl_available",
    "isend",
    "monitored_barrier",
    "new_group",
    "new_subgroups",
    "new_subgroups_by_enumeration",
    "recv",
    "recv_object_list",
    "reduce",
    "reduce_op",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "scatter",
    "scatter_object_list",
    "send",
    "send_object_list",
    "split_group",
    "supports_complex",
]
_MPI_AVAILABLE = ...
_NCCL_AVAILABLE = ...
_GLOO_AVAILABLE = ...
_UCC_AVAILABLE = ...
_XCCL_AVAILABLE = ...
_pickler = pickle.Pickler
_unpickler = pickle.Unpickler
__all__ += ["ProcessGroupMPI"]
__all__ += ["ProcessGroupNCCL"]
__all__ += ["ProcessGroupGloo"]
__all__ += ["ProcessGroupUCC"]
__all__ += ["ProcessGroupXCCL"]
logger = ...
PG_WRAPPER_STORE_PREFIX = ...

def supports_complex(reduceOp: ReduceOp) -> bool: ...

class Backend(str):
    UNDEFINED = ...
    GLOO = ...
    NCCL = ...
    UCC = ...
    MPI = ...
    XCCL = ...
    _BackendPlugin = ...
    _plugins: dict[str, _BackendPlugin] = ...
    backend_list = ...
    default_device_backend_map: dict[str, str] = ...
    backend_capability: dict[str, list[str]] = ...
    backend_type_map: dict[str, ProcessGroup.BackendType] = ...
    def __new__(cls, name: str) -> str | Any: ...
    @classmethod
    def register_backend(
        cls, name, func, extended_api=..., devices: str | list[str] | None = ...
    ) -> None: ...

class BackendConfig:
    def __init__(self, backend: Backend) -> None: ...
    def get_device_backend_map(self) -> dict[str, Backend]: ...

class _reduce_op:
    def __init__(self) -> None: ...
    @deprecated(
        "`torch.distributed.reduce_op` is deprecated, please use `torch.distributed.ReduceOp` instead",
        category=FutureWarning,
    )
    def __getattribute__(self, key) -> Any: ...

reduce_op = ...

class P2POp:
    def __init__(
        self,
        op: Callable,
        tensor: torch.Tensor,
        peer: int | None = ...,
        group: ProcessGroup | None = ...,
        tag: int = ...,
        group_peer: int | None = ...,
    ) -> None: ...
    def __new__(
        cls,
        op: Callable,
        tensor: torch.Tensor,
        peer: int | None = ...,
        group: ProcessGroup | None = ...,
        tag: int = ...,
        group_peer: int | None = ...,
    ) -> Self: ...

class _CollOp:
    def __init__(
        self,
        op: Callable,
        tensor: torch.Tensor,
        dst_tensor: torch.Tensor | None = ...,
        redop: ReduceOp | None = ...,
        root: int | None = ...,
    ) -> None: ...

_pg_map: dict[ProcessGroup, tuple[str, Store]] = ...
_pg_names: dict[ProcessGroup, str] = ...
_pg_group_ranks: dict[ProcessGroup, dict[int, int]] = ...
_pg_backend_config: dict[ProcessGroup, str] = ...
_group_count = ...
_tags_to_pg: dict[str, list[ProcessGroup]] = ...
_pg_to_tag: dict[ProcessGroup, str] = ...
_backend: str | None = ...

class _World:
    def __init__(self) -> None: ...
    @property
    def default_pg(self) -> ProcessGroup | None: ...
    @default_pg.setter
    def default_pg(self, value) -> None: ...
    @property
    def pg_map(self) -> dict[ProcessGroup, tuple[str, Store]]: ...
    @property
    def pg_names(self) -> dict[ProcessGroup, str]: ...
    @property
    def pg_group_ranks(self) -> dict[ProcessGroup, dict[int, int]]: ...
    @property
    def pg_backend_config(self) -> dict[ProcessGroup, str]: ...
    @property
    def group_count(self) -> int: ...
    @group_count.setter
    def group_count(self, value: int) -> None: ...
    @property
    def tags_to_pg(self) -> dict[str, list[ProcessGroup]]: ...
    @property
    def pg_to_tag(self) -> dict[ProcessGroup, str]: ...
    @property
    def pg_coalesce_state(self) -> dict[ProcessGroup, list[_CollOp]]: ...
    @property
    def pg_config_info(self) -> list[dict[str, Any]]: ...

_world = ...

class _WorldMeta(type):
    @property
    def WORLD(cls) -> ProcessGroup | None: ...
    @WORLD.setter
    def WORLD(cls, pg: ProcessGroup | None) -> None: ...

class group(metaclass=_WorldMeta): ...

class GroupMember(metaclass=_WorldMeta):
    NON_GROUP_MEMBER = ...

_default_pg_init_method: str | None = ...
STORE_BASED_BARRIER_PREFIX = ...

def get_group_rank(group: ProcessGroup, global_rank: int) -> int: ...
def get_global_rank(group: ProcessGroup, group_rank: int) -> int: ...
def get_process_group_ranks(group: ProcessGroup | None) -> list[int]: ...
def is_mpi_available() -> bool: ...
def is_nccl_available() -> bool: ...
def is_gloo_available() -> bool: ...
def is_ucc_available() -> bool: ...
def is_xccl_available() -> bool: ...
def is_backend_available(backend: str) -> bool: ...
def is_initialized() -> bool: ...
def is_torchelastic_launched() -> bool: ...
def get_backend_config(group: ProcessGroup | None = ...) -> str: ...
def get_backend(group: ProcessGroup | None = ...) -> Backend: ...
def get_default_backend_for_device(device: str | torch.device) -> str: ...
def get_pg_count() -> int: ...
def get_node_local_rank(fallback_rank: int | None = ...) -> int: ...
@_exception_logger
@_time_logger
def init_process_group(
    backend: str | None = ...,
    init_method: str | None = ...,
    timeout: timedelta | None = ...,
    world_size: int = ...,
    rank: int = ...,
    store: Store | None = ...,
    group_name: str = ...,
    pg_options: Any | None = ...,
    device_id: torch.device | int | None = ...,
) -> None: ...
def destroy_process_group(group: ProcessGroup | None = ...) -> None: ...
def get_rank(group: ProcessGroup | None = ...) -> int: ...
def get_world_size(group: ProcessGroup | None = ...) -> int: ...
def isend(
    tensor: torch.Tensor,
    dst: int | None = ...,
    group: ProcessGroup | None = ...,
    tag: int = ...,
    group_dst: int | None = ...,
) -> Work | None: ...
def irecv(
    tensor: torch.Tensor,
    src: int | None = ...,
    group: ProcessGroup | None = ...,
    tag: int = ...,
    group_src: int | None = ...,
) -> Work | None: ...
@_exception_logger
def send(
    tensor: torch.Tensor,
    dst: int | None = ...,
    group: ProcessGroup | None = ...,
    tag: int = ...,
    group_dst: int | None = ...,
) -> None: ...
@_exception_logger
def recv(
    tensor: torch.Tensor,
    src: int | None = ...,
    group: ProcessGroup | None = ...,
    tag: int = ...,
    group_src: int | None = ...,
) -> int: ...

class _IllegalWork(Work):
    def __getattribute__(self, name) -> None: ...

class _CoalescingManager:
    def __init__(self) -> None: ...
    def append(self, work: Work | None = ...) -> None: ...
    def wait(self) -> None: ...

class _TimeEstimator:
    def __init__(self) -> None: ...

def batch_isend_irecv(p2p_op_list: list[P2POp]) -> list[Work]: ...
@_exception_logger
def broadcast(
    tensor: torch.Tensor,
    src: int | None = ...,
    group: ProcessGroup | None = ...,
    async_op: bool = ...,
    group_src: int | None = ...,
) -> Work | None: ...
def all_reduce(
    tensor: Tensor,
    op: ReduceOp | ReduceOp.RedOpType = ...,
    group: ProcessGroup | None = ...,
    async_op: bool = ...,
) -> Work | None: ...
def all_reduce_coalesced(
    tensors, op=..., group=..., async_op=...
) -> Future[Any] | None: ...
@_exception_logger
def reduce(
    tensor: torch.Tensor,
    dst: int | None = ...,
    op=...,
    group: ProcessGroup | None = ...,
    async_op: bool = ...,
    group_dst: int | None = ...,
) -> Work | None: ...
@_exception_logger
def all_gather_object(
    object_list: list[Any], obj: Any, group: ProcessGroup | None = ...
) -> None: ...
@_exception_logger
def gather_object(
    obj: Any,
    object_gather_list: list[Any] | None = ...,
    dst: int | None = ...,
    group: ProcessGroup | None = ...,
    group_dst: int | None = ...,
) -> None: ...
@_exception_logger
def send_object_list(
    object_list: list[Any],
    dst: int | None = ...,
    group: ProcessGroup | None = ...,
    device: torch.device | None = ...,
    group_dst: int | None = ...,
    use_batch: bool = ...,
) -> None: ...
@_exception_logger
def recv_object_list(
    object_list: list[Any],
    src: int | None = ...,
    group: ProcessGroup | None = ...,
    device: torch.device | None = ...,
    group_src: int | None = ...,
    use_batch: bool = ...,
) -> int: ...
@_exception_logger
def broadcast_object_list(
    object_list: list[Any],
    src: int | None = ...,
    group: ProcessGroup | None = ...,
    device: torch.device | None = ...,
    group_src: int | None = ...,
) -> None: ...
@_exception_logger
def scatter_object_list(
    scatter_object_output_list: list[Any],
    scatter_object_input_list: list[Any] | None = ...,
    src: int | None = ...,
    group: ProcessGroup | None = ...,
    group_src: int | None = ...,
) -> None: ...
def all_gather(
    tensor_list: list[Tensor],
    tensor: Tensor,
    group: ProcessGroup | None = ...,
    async_op: bool = ...,
) -> Work | None: ...
def all_gather_into_tensor(
    output_tensor: Tensor,
    input_tensor: Tensor,
    group: ProcessGroup | None = ...,
    async_op: bool = ...,
) -> Work | None: ...
def all_gather_coalesced(
    output_tensor_lists, input_tensor_list, group=..., async_op=...
) -> Future[Any] | None: ...
@_exception_logger
def gather(
    tensor: torch.Tensor,
    gather_list: list[torch.Tensor] | None = ...,
    dst: int | None = ...,
    group: ProcessGroup | None = ...,
    async_op: bool = ...,
    group_dst: int | None = ...,
) -> Work | None: ...
@_exception_logger
def scatter(
    tensor: torch.Tensor,
    scatter_list: list[torch.Tensor] | None = ...,
    src: int | None = ...,
    group: ProcessGroup | None = ...,
    async_op: bool = ...,
    group_src: int | None = ...,
) -> Work | None: ...
@_exception_logger
def reduce_scatter(
    output, input_list, op=..., group=..., async_op=...
) -> Work | None: ...
def reduce_scatter_tensor(
    output: Tensor,
    input: Tensor,
    op: ReduceOp = ...,
    group: ProcessGroup | None = ...,
    async_op: bool = ...,
) -> Work | None: ...
def all_to_all_single(
    output,
    input,
    output_split_sizes=...,
    input_split_sizes=...,
    group=...,
    async_op=...,
) -> Work | None: ...
def all_to_all(
    output_tensor_list: list[Tensor],
    input_tensor_list: list[Tensor],
    group: ProcessGroup | None = ...,
    async_op: bool = ...,
) -> Work | None: ...
def barrier(
    group: ProcessGroup | None = ..., async_op=..., device_ids=...
) -> Work | None: ...
def monitored_barrier(
    group: ProcessGroup | None = ..., timeout=..., wait_all_ranks=...
) -> None: ...
@_time_logger
def split_group(
    parent_pg: ProcessGroup | None = ...,
    split_ranks: list | None = ...,
    timeout: timedelta | None = ...,
    pg_options: Any | None = ...,
    group_desc: str | None = ...,
) -> ProcessGroup | None: ...
@_time_logger
def new_group(
    ranks=...,
    timeout=...,
    backend=...,
    pg_options=...,
    use_local_synchronization=...,
    group_desc=...,
    device_id: torch.device | None = ...,
) -> None: ...
def new_subgroups(
    group_size=..., group=..., timeout=..., backend=..., pg_options=..., group_desc=...
) -> tuple[Any | None, list[Any]]: ...
def new_subgroups_by_enumeration(
    ranks_per_subgroup_list, timeout=..., backend=..., pg_options=..., group_desc=...
) -> tuple[Any | None, list[Any]]: ...
