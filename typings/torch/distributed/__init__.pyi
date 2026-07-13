from datetime import timedelta as timedelta

import pdb  # noqa: T100 -- mirrors upstream debugger import
import typing

from torch._C._distributed_c10d import (
    _DEFAULT_FIRST_BUCKET_BYTES as _DEFAULT_FIRST_BUCKET_BYTES,
    BuiltinCommHookType as BuiltinCommHookType,
    DebugLevel as DebugLevel,
    FileStore as FileStore,
    GradBucket as GradBucket,
    Logger as Logger,
    PrefixStore as PrefixStore,
    ProcessGroup as ProcessGroup,
    Reducer as Reducer,
    Store as Store,
    TCPStore as TCPStore,
    _broadcast_coalesced as _broadcast_coalesced,
    _compute_bucket_assignment_by_size as _compute_bucket_assignment_by_size,
    _ControlCollectives as _ControlCollectives,
    _make_nccl_premul_sum as _make_nccl_premul_sum,
    _register_builtin_comm_hook as _register_builtin_comm_hook,
    _register_comm_hook as _register_comm_hook,
    _StoreCollectives as _StoreCollectives,
    _test_python_store as _test_python_store,
    _verify_params_across_processes as _verify_params_across_processes,
    get_debug_level as get_debug_level,
    set_debug_level as set_debug_level,
    set_debug_level_from_env as set_debug_level_from_env,
)

import torch

from .device_mesh import (
    DeviceMesh as DeviceMesh,
    init_device_mesh as init_device_mesh,
)
from .distributed_c10d import *
from .distributed_c10d import (
    _all_gather_base as _all_gather_base,
    _coalescing_manager as _coalescing_manager,
    _CoalescingManager as _CoalescingManager,
    _create_process_group_wrapper as _create_process_group_wrapper,
    _get_process_group_name as _get_process_group_name,
    _rank_not_in_group as _rank_not_in_group,
    _reduce_scatter_base as _reduce_scatter_base,
    _time_estimator as _time_estimator,
    get_node_local_rank as get_node_local_rank,
)
from .remote_device import _remote_device as _remote_device
from .rendezvous import (
    _create_store_from_options as _create_store_from_options,
    register_rendezvous_handler as register_rendezvous_handler,
    rendezvous as rendezvous,
)

log = ...

def is_available() -> bool: ...

DistError = torch._C._DistError
DistBackendError = torch._C._DistBackendError
DistNetworkError = torch._C._DistNetworkError
DistStoreError = torch._C._DistStoreError
QueueEmptyError = torch._C._DistQueueEmptyError
if is_available():
    class _DistributedPdb(pdb.Pdb):
        def interaction(self, *args, **kwargs) -> None: ...

    _breakpoint_cache: dict[int, typing.Any] = ...
    def breakpoint(rank: int = ..., skip: int = ..., timeout_s=...) -> None: ...

else:
    class _ProcessGroupStub: ...

default_pg_timeout: timedelta
