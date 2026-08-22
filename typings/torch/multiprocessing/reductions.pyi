from collections.abc import Callable
from typing import Any, Self

from torch import Size, Tensor

class StorageWeakRef:
    __slots__ = ...
    def __init__(self, storage) -> None: ...
    @classmethod
    def from_weakref(cls, cdata) -> Self: ...
    def expired(self): ...
    def __del__(self) -> None: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other) -> bool: ...

class SharedCache(dict):
    def __init__(self) -> None: ...
    def get(self, key) -> None: ...
    def __setitem__(self, key, storage_ref) -> None: ...
    def free_dead_references(self) -> None: ...

shared_cache = ...

def rebuild_event(device, handle) -> _CudaEventBase: ...
def reduce_event(event) -> tuple[Callable[..., _CudaEventBase], tuple[Any, Any]]: ...
def rebuild_tensor(cls, storage, metadata) -> Parameter: ...
def rebuild_meta_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    dtype,
    storage_size_bytes,
    requires_grad,
) -> Parameter: ...
def rebuild_cuda_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    storage_cls,
    dtype,
    storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    requires_grad,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
) -> Parameter: ...
def reduce_tensor(
    tensor,
) -> (
    tuple[
        Callable[..., Tensor],
        tuple[
            Any | Callable[..., Tensor] | Callable[..., Parameter | Any],
            Any
            | tuple[Any, Any, Any, Any, Any | Size, bool | Any]
            | tuple[Any, Any, Any, Any, Any, Any, Any | Size, Any | layout]
            | tuple[
                type[Any | NestedTensor],
                Size | Any,
                tuple[int, ...] | Any,
                int | SymInt | Any,
                type[TypedStorage | Any],
                Any | dtype,
                Any,
                Any,
                Any,
                Any,
                Any | bool,
                Any,
                Any,
                Any,
                Any,
            ]
            | tuple[
                type[Any | NestedTensor],
                Size | Any,
                tuple[int, ...] | Any,
                int | SymInt | Any,
                Any | dtype,
                int | Any,
                Any | bool,
            ]
            | tuple[
                type[Any | NestedTensor],
                TypedStorage | Any,
                tuple[
                    int | SymInt | Any, Size | Any, tuple[int, ...] | Any, Any | bool
                ],
            ],
            Any,
            Any,
            Any | Callable[..., Tensor] | Callable[..., Parameter | Any],
            Any
            | tuple[Any, Any, Any, Any, Any | Size, bool | Any]
            | tuple[Any, Any, Any, Any, Any, Any, Any | Size, Any | layout]
            | tuple[
                type[Any | NestedTensor],
                Size | Any,
                tuple[int, ...] | Any,
                int | SymInt | Any,
                type[TypedStorage | Any],
                Any | dtype,
                Any,
                Any,
                Any,
                Any,
                Any | bool,
                Any,
                Any,
                Any,
                Any,
            ]
            | tuple[
                type[Any | NestedTensor],
                Size | Any,
                tuple[int, ...] | Any,
                int | SymInt | Any,
                Any | dtype,
                int | Any,
                Any | bool,
            ]
            | tuple[
                type[Any | NestedTensor],
                TypedStorage | Any,
                tuple[
                    int | SymInt | Any, Size | Any, tuple[int, ...] | Any, Any | bool
                ],
            ],
            Any | Callable[..., Tensor] | Callable[..., Parameter | Any],
            Any
            | tuple[Any, Any, Any, Any, Any | Size, bool | Any]
            | tuple[Any, Any, Any, Any, Any, Any, Any | Size, Any | layout]
            | tuple[
                type[Any | NestedTensor],
                Size | Any,
                tuple[int, ...] | Any,
                int | SymInt | Any,
                type[TypedStorage | Any],
                Any | dtype,
                Any,
                Any,
                Any,
                Any,
                Any | bool,
                Any,
                Any,
                Any,
                Any,
            ]
            | tuple[
                type[Any | NestedTensor],
                Size | Any,
                tuple[int, ...] | Any,
                int | SymInt | Any,
                Any | dtype,
                int | Any,
                Any | bool,
            ]
            | tuple[
                type[Any | NestedTensor],
                TypedStorage | Any,
                tuple[
                    int | SymInt | Any, Size | Any, tuple[int, ...] | Any, Any | bool
                ],
            ],
        ],
    ]
    | tuple[Callable[..., Tensor], tuple[Any, Any, Any, Any, Any | Size, bool | Any]]
    | tuple[
        Callable[..., Tensor],
        tuple[Any, Any, Any, Any, Any, Any, Any | Size, Any | layout],
    ]
    | tuple[
        Callable[..., Parameter | Any],
        tuple[
            type[Any | NestedTensor],
            Size | Any,
            tuple[int, ...] | Any,
            int | SymInt | Any,
            type[TypedStorage | Any],
            Any | dtype,
            Any,
            Any,
            Any,
            Any,
            Any | bool,
            Any,
            Any,
            Any,
            Any,
        ],
    ]
    | tuple[
        Callable[..., Parameter | Any],
        tuple[
            type[Any | NestedTensor],
            Size | Any,
            tuple[int, ...] | Any,
            int | SymInt | Any,
            Any | dtype,
            int | Any,
            Any | bool,
        ],
    ]
    | tuple[
        Callable[..., Parameter | Any],
        tuple[
            type[Any | NestedTensor],
            TypedStorage | Any,
            tuple[int | SymInt | Any, Size | Any, tuple[int, ...] | Any, Any | bool],
        ],
    ]
): ...
def rebuild_nested_tensor(
    rebuild_buffer_func,
    rebuild_buffer_args,
    rebuild_sizes_func,
    rebuild_sizes_args,
    rebuild_strides_func,
    rebuild_strides_args,
    rebuild_offsets_func,
    rebuild_offsets_args,
) -> Tensor: ...
def reduce_nested_tensor(
    nt,
) -> tuple[
    Callable[..., Tensor],
    tuple[
        Callable[..., Tensor] | Callable[..., Parameter | Any],
        Any
        | tuple[Any, Any, Any, Any, Any | Size, bool | Any]
        | tuple[Any, Any, Any, Any, Any, Any, Any | Size, Any | layout]
        | tuple[
            type[Any | NestedTensor],
            Size | Any,
            tuple[int, ...] | Any,
            int | SymInt | Any,
            type[TypedStorage | Any],
            Any | dtype,
            Any,
            Any,
            Any,
            Any,
            Any | bool,
            Any,
            Any,
            Any,
            Any,
        ]
        | tuple[
            type[Any | NestedTensor],
            Size | Any,
            tuple[int, ...] | Any,
            int | SymInt | Any,
            Any | dtype,
            int | Any,
            Any | bool,
        ]
        | tuple[
            type[Any | NestedTensor],
            TypedStorage | Any,
            tuple[int | SymInt | Any, Size | Any, tuple[int, ...] | Any, Any | bool],
        ],
        Any,
        Any,
        Any | Callable[..., Tensor] | Callable[..., Parameter | Any],
        Any
        | tuple[Any, Any, Any, Any, Any | Size, bool | Any]
        | tuple[Any, Any, Any, Any, Any, Any, Any | Size, Any | layout]
        | tuple[
            type[Any | NestedTensor],
            Size | Any,
            tuple[int, ...] | Any,
            int | SymInt | Any,
            type[TypedStorage | Any],
            Any | dtype,
            Any,
            Any,
            Any,
            Any,
            Any | bool,
            Any,
            Any,
            Any,
            Any,
        ]
        | tuple[
            type[Any | NestedTensor],
            Size | Any,
            tuple[int, ...] | Any,
            int | SymInt | Any,
            Any | dtype,
            int | Any,
            Any | bool,
        ]
        | tuple[
            type[Any | NestedTensor],
            TypedStorage | Any,
            tuple[int | SymInt | Any, Size | Any, tuple[int, ...] | Any, Any | bool],
        ],
        Any | Callable[..., Tensor] | Callable[..., Parameter | Any],
        Any
        | tuple[Any, Any, Any, Any, Any | Size, bool | Any]
        | tuple[Any, Any, Any, Any, Any, Any, Any | Size, Any | layout]
        | tuple[
            type[Any | NestedTensor],
            Size | Any,
            tuple[int, ...] | Any,
            int | SymInt | Any,
            type[TypedStorage | Any],
            Any | dtype,
            Any,
            Any,
            Any,
            Any,
            Any | bool,
            Any,
            Any,
            Any,
            Any,
        ]
        | tuple[
            type[Any | NestedTensor],
            Size | Any,
            tuple[int, ...] | Any,
            int | SymInt | Any,
            Any | dtype,
            int | Any,
            Any | bool,
        ]
        | tuple[
            type[Any | NestedTensor],
            TypedStorage | Any,
            tuple[int | SymInt | Any, Size | Any, tuple[int, ...] | Any, Any | bool],
        ],
    ],
]: ...
def rebuild_sparse_coo_tensor(
    rebuild_indices_func,
    rebuild_indices_args,
    rebuild_values_func,
    rebuild_values_args,
    shape,
    is_coalesced,
) -> Tensor: ...
def rebuild_sparse_compressed_tensor(
    rebuild_compressed_indices_func,
    rebuild_compressed_indices_args,
    rebuild_plain_indices_func,
    rebuild_plain_indices_args,
    rebuild_values_func,
    rebuild_values_args,
    shape,
    layout,
) -> Tensor: ...
def reduce_sparse_tensor(
    sparse,
) -> (
    tuple[Callable[..., Tensor], tuple[Any, Any, Any, Any, Any, Any]]
    | tuple[Callable[..., Tensor], tuple[Any, Any, Any, Any, Any, Any, Any, Any]]
): ...
def fd_id(fd) -> tuple[int, int]: ...
def storage_from_cache(cls, key) -> UntypedStorage | None: ...
def rebuild_storage_fd(cls, df, size) -> UntypedStorage: ...
def rebuild_storage_filename(
    cls, manager, handle, size, dtype=...
) -> _StorageBase | TypedStorage: ...
def rebuild_storage_empty(cls): ...
def rebuild_typed_storage(storage, dtype) -> TypedStorage: ...
def reduce_typed_storage(
    storage,
) -> tuple[Callable[..., TypedStorage], tuple[Any, Any]]: ...
def rebuild_typed_storage_child(storage, storage_type): ...
def reduce_typed_storage_child(
    storage,
) -> tuple[Callable[..., Any], tuple[Any, type[Any]]]: ...
def reduce_storage(
    storage,
) -> (
    tuple[Callable[..., Any], tuple[type[Any]]]
    | tuple[
        Callable[..., _StorageBase | TypedStorage]
        | Callable[..., UntypedStorage | Any],
        Any | tuple[type[TypedStorage | Any], Any, Any],
    ]
): ...
def init_reductions() -> None: ...
