from collections.abc import Generator as Generator
from datetime import timedelta as timedelta
from urllib.parse import urlparse as urlparse

from torch._C._distributed_c10d import Store as Store
from torch._C._distributed_rpc import (
    _DEFAULT_INIT_METHOD as _DEFAULT_INIT_METHOD,
    _DEFAULT_RPC_TIMEOUT_SEC as _DEFAULT_RPC_TIMEOUT_SEC,
    _UNSET_RPC_TIMEOUT as _UNSET_RPC_TIMEOUT,
    PyRRef as PyRRef,
    RemoteProfilerManager as RemoteProfilerManager,
    RpcAgent as RpcAgent,
    RpcBackendOptions as RpcBackendOptions,
    WorkerInfo as WorkerInfo,
    _cleanup_python_rpc_handler as _cleanup_python_rpc_handler,
    _delete_all_user_and_unforked_owner_rrefs as _delete_all_user_and_unforked_owner_rrefs,
    _destroy_rref_context as _destroy_rref_context,
    _disable_jit_rref_pickle as _disable_jit_rref_pickle,
    _disable_server_process_global_profiler as _disable_server_process_global_profiler,
    _enable_jit_rref_pickle as _enable_jit_rref_pickle,
    _enable_server_process_global_profiler as _enable_server_process_global_profiler,
    _get_current_rpc_agent as _get_current_rpc_agent,
    _invoke_remote_builtin as _invoke_remote_builtin,
    _invoke_remote_python_udf as _invoke_remote_python_udf,
    _invoke_remote_torchscript as _invoke_remote_torchscript,
    _invoke_rpc_builtin as _invoke_rpc_builtin,
    _invoke_rpc_python_udf as _invoke_rpc_python_udf,
    _invoke_rpc_torchscript as _invoke_rpc_torchscript,
    _is_current_rpc_agent_set as _is_current_rpc_agent_set,
    _reset_current_rpc_agent as _reset_current_rpc_agent,
    _rref_context_get_debug_info as _rref_context_get_debug_info,
    _set_and_start_rpc_agent as _set_and_start_rpc_agent,
    _set_profiler_node_id as _set_profiler_node_id,
    _set_rpc_timeout as _set_rpc_timeout,
    enable_gil_profiling as enable_gil_profiling,
    get_rpc_timeout as get_rpc_timeout,
)

from . import (
    api as api,
    backend_registry as backend_registry,
    functions as functions,
)
from .api import *
from .backend_registry import BackendType as BackendType
from .options import TensorPipeRpcBackendOptions as TensorPipeRpcBackendOptions
from .server_process_global_profiler import (
    _server_process_global_profile as _server_process_global_profile,
)

__all__ = ["is_available"]
logger = ...
_init_counter = ...
_init_counter_lock = ...

def is_available() -> bool: ...

if is_available():
    _is_tensorpipe_available = ...
    rendezvous_iterator: Generator[tuple[Store, int, int]]
    def init_rpc(
        name, backend=..., rank=..., world_size=..., rpc_backend_options=...
    ) -> None: ...
