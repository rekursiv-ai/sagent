from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self, TypeGuard, TypeVar
from weakref import ReferenceType

import atexit
import contextlib
import functools
import threading
import types
import weakref

from torch import SymInt, Tensor
from torch._guards import Source
from torch._ops import OpOverload
from torch._subclasses.meta_utils import MetaConverter
from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymbolicContext
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.types import IntLikeType
from torch.utils._backport_slots import dataclass_slots
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import PyTree, TreeSpec
from torch.utils._stats import count

import torch

from ._fake_tensor_utils import _PySymInputStub, _SymIntOutputStub

log = ...
hc_log = ...
not_implemented_log = ...
DimList = list
pytree = ...
T = TypeVar("T")
aten = ...
CONSTANT_NUMEL_LIMIT = ...
RECURSION_COUNT = ...

class IncrementRecursionCount:
    def __init__(self) -> None: ...
    def __del__(self) -> None: ...

@dataclass
class UnsupportedFakeTensorException(RuntimeError):
    reason: str

@dataclass
class DynamicOutputShapeException(RuntimeError):
    func: OpOverload

@dataclass
class DataDependentOutputException(RuntimeError):
    func: OpOverload

@dataclass
class UnsupportedOperatorException(RuntimeError):
    func: OpOverload

@dataclass
class UnsupportedMutationAliasingException(RuntimeError):
    reason: str

@dataclass
class MetadataMismatchError(RuntimeError):
    reason: str

class FakeTensorTLS(threading.local):
    allow_non_fake_inputs_override: bool | None
    def __init__(self) -> None: ...

fake_tensor_tls = ...

def ordered_set(*items: T) -> dict[T, Literal[True]]: ...
@contextlib.contextmanager
def unset_fake_temporarily() -> Generator[TorchDispatchMode | None]: ...
@contextlib.contextmanager
def disable_fake_tensor_cache(fake_mode: FakeTensorMode) -> Generator[None]: ...
def get_plain_tensors(
    subclass: Tensor, *, out: list[Tensor | int | SymInt]
) -> list[Tensor | int | SymInt]: ...
def is_fake(x: object) -> TypeGuard[Tensor]: ...
def maybe_get_fake_mode(t: object) -> FakeTensorMode | None: ...
@functools.cache
def get_schema_info(func: OpOverload) -> torch._C._SchemaInfo: ...
@functools.cache
def torch_decomp_decompositions(func: OpOverload) -> bool: ...
def tree_flatten_only(ty: type[T], tree: PyTree) -> list[T]: ...

class FakeTensorConverter:
    @property
    def tensor_memo(self) -> weakref.WeakValueDictionary: ...

    meta_converter: MetaConverter
    constant_storage_mapping: dict[StorageWeakRef, list[ReferenceType]]
    export: bool
    def __init__(self, *, copy_data: bool = ..., export: bool = ...) -> None: ...
    def add_constant_storage_mapping(self, fake_tensor: FakeTensor) -> None: ...
    def invalidate_constant_aliases(self, tensor: Tensor) -> None: ...
    def set_tensor_memo(self, t: Tensor, v: FakeTensor) -> None: ...
    def from_real_tensor(
        self,
        fake_mode: FakeTensorMode,
        t: Tensor,
        make_constant: bool = ...,
        shape_env: ShapeEnv | None = ...,
        *,
        source: Source | None = ...,
        symbolic_context: SymbolicContext | None = ...,
        trace: bool = ...,
    ) -> FakeTensor: ...
    def from_meta_and_device(
        self,
        fake_mode: FakeTensorMode,
        t: Tensor,
        device: torch.device,
        pytype: type[torch.Tensor] | None = ...,
        dispatch_keys: torch.DispatchKeySet | None = ...,
    ) -> FakeTensor: ...

@functools.cache
def init_gpu_context(device: torch.device) -> None: ...
@contextlib.contextmanager
def in_kernel_invocation_manager(fake_mode: FakeTensorMode) -> Generator[None]: ...
def should_allow_numbers_as_tensors(func: OpOverload) -> bool: ...

class FakeTensorConfig:
    debug = ...

class SymNumberMemoDescriptor:
    _name: str
    _is_nested_int: bool
    def __init__(self, *, is_nested_int: bool = ...) -> None: ...
    def __set_name__(self, owner: str, name: str) -> None: ...
    def __get__(
        self, obj: FakeTensor, objtype: type[FakeTensor] | None = ...
    ) -> torch.SymInt | torch.SymFloat | None: ...
    def __set__(
        self, obj: FakeTensor, value: torch.SymInt | torch.SymFloat | None
    ) -> None: ...

class FakeTensor(Tensor):
    fake_device: torch.device
    fake_mode: FakeTensorMode
    constant: Tensor | None
    real_tensor: Tensor | None
    nonzero_memo = ...
    item_memo = ...
    unique_memo = ...
    unique_consecutive_memo = ...
    nested_int_memo = ...
    pytype: type[Tensor] | None
    dispatch_keys: torch.DispatchKeySet | None
    _mode_key = ...
    @property
    def device(self) -> torch.device: ...
    @device.setter
    def device(self, _: torch.device) -> None: ...
    @property
    def names(self) -> list[str]: ...
    @names.setter
    def names(self, _: list[str]) -> None: ...
    @staticmethod
    def __new__(
        cls,
        fake_mode: FakeTensorMode,
        elem: Tensor,
        device: torch.device,
        constant: Tensor | None = ...,
        real_tensor: Tensor | None = ...,
        pytype: type[Tensor] | None = ...,
        dispatch_keys: torch.DispatchKeySet | None = ...,
    ) -> Self: ...
    def __init__(self, *args: object, **kwargs: object) -> None: ...
    @staticmethod
    def from_tensor(t: Tensor, fake_mode: FakeTensorMode) -> FakeTensor: ...
    @classmethod
    @count
    def __torch_dispatch__(
        cls,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object] = ...,
        kwargs: Mapping[str, object] = ...,
    ) -> object: ...
    def get_nested_int(self, *, coeff: int | torch.SymInt = ...) -> torch.SymInt: ...
    def tolist(self) -> Any: ...

type _MetadataIntLike = IntLikeType | _PySymInputStub | _SymIntOutputStub

@dataclass_slots
@dataclass
class TensorMetadata:
    dtype: torch.dtype
    shape: tuple[_MetadataIntLike, ...]
    stride: tuple[_MetadataIntLike, ...]
    device: torch.device
    layout: torch.layout
    memory_format: torch.memory_format | None
    storage_offset: _MetadataIntLike
    storage_bytes: _MetadataIntLike | None
    requires_grad: bool
    is_quantized: bool
    is_conj: bool
    is_neg: bool
    is_inference: bool
    is_sparse: bool
    is_coalesced: bool | None
    dense_dim: int | None
    sparse_dim: int | None

def extract_tensor_metadata(t: Tensor) -> TensorMetadata: ...

@dataclass_slots
@dataclass
class _DispatchCacheKey:
    key: tuple[object, ...]
    hashvalue: int
    def __init__(self, tup: tuple[object, ...]) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def strip_shape_env(self) -> None: ...

class SingletonConstant: ...

@dataclass_slots
@dataclass(frozen=True)
class _DispatchCacheEntryOutputInfo:
    inplace_idx: int | None
    metadata: TensorMetadata | None
    view_idx: int | None
    constant_value: Any | None = ...

@dataclass_slots
@dataclass(frozen=True)
class _DispatchCacheValidEntry:
    output_infos: tuple[_DispatchCacheEntryOutputInfo]
    is_output_tuple: bool = ...

@dataclass_slots
@dataclass(frozen=True)
class _DispatchCacheBypassEntry:
    reason: str

type _DispatchCacheEntry = _DispatchCacheValidEntry | _DispatchCacheBypassEntry

@dataclass_slots
@dataclass(frozen=True)
class _BypassDispatchCache(Exception):
    reason: str

@dataclass_slots
@dataclass(frozen=True)
class DispatchCacheInfo:
    hits: int
    misses: int
    bypasses: dict[str, int]
    size: int

class FakeTensorMode(TorchDispatchMode):
    cache: dict[_DispatchCacheKey, _DispatchCacheEntry] = ...
    cache_hits: int = ...
    cache_misses: int = ...
    cache_bypasses: dict[str, int] = ...
    epoch: int = ...
    in_kernel_invocation: bool = ...
    static_shapes: bool
    shape_env: ShapeEnv | None
    _stack: str | None
    allow_meta: bool
    nt_tensor_id_counter: int = ...
    nt_tensor_id_initial_count: int = ...
    def __init__(
        self,
        *,
        allow_fallback_kernels: bool = ...,
        allow_non_fake_inputs: bool = ...,
        shape_env: ShapeEnv | None = ...,
        static_shapes: bool | None = ...,
        export: bool = ...,
    ) -> None: ...
    def reset_nt_tensor_id_counter(self) -> None: ...
    def is_our_fake(self, t: object) -> TypeGuard[FakeTensor]: ...
    @property
    def avoid_device_init(self) -> bool: ...
    @property
    def stack(self) -> str: ...
    @count
    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object] = ...,
        kwargs: Mapping[str, object] = ...,
    ) -> object: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        a: type[BaseException] | None,
        b: BaseException | None,
        c: types.TracebackType | None,
    ) -> None: ...
    @classmethod
    def is_infra_mode(cls) -> bool: ...
    @classmethod
    def cache_info(cls) -> DispatchCacheInfo: ...
    @classmethod
    def cache_clear(cls) -> None: ...
    def dispatch(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object] = ...,
        kwargs: Mapping[str, object] = ...,
    ) -> object: ...

    _can_run_unsafe_fallback_allowed_namespaces = ...
    def can_run_unsafe_fallback(self, func: OpOverload) -> bool: ...
    def validate_and_convert_non_fake_tensors(
        self,
        func: OpOverload,
        converter: FakeTensorConverter,
        flat_args: Sequence[object],
        args_spec: TreeSpec,
    ) -> tuple[list[object], list[FakeTensor]]: ...
    def wrap_meta_outputs_with_default_device_logic(
        self,
        r: object,
        func: OpOverload,
        flat_args: Sequence[object],
        device: torch.device,
    ) -> PyTree: ...
    def create_symbolic_nested_int(
        self, *, nt_tensor_id: int | None = ...
    ) -> torch.SymInt: ...

    _cpp_meta_supports_symint = ...
    _view_fake_tensor_impl_ops = ...
    def cpp_meta_supports_symint(self, func: OpOverload) -> bool: ...

    lift_fns = ...
    def may_turn_const(self, t: Tensor) -> bool: ...
    def invalidate_written_to_constants(
        self,
        func: OpOverload,
        flat_arg_fake_tensors: Sequence[FakeTensor],
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> None: ...
    def from_tensor(
        self,
        tensor: Tensor,
        *,
        static_shapes: bool | None = ...,
        source: Source | None = ...,
        symbolic_context: SymbolicContext | None = ...,
        trace: bool = ...,
    ) -> FakeTensor: ...

_StoragePointer = object

def run_fallback_kernel(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    flat_args: Sequence[object],
    args_spec: PyTree,
    orig_not_implemented_exception: RuntimeError,
) -> FakeTensor: ...

class FakeCopyMode(TorchFunctionMode):
    def __init__(self, fake_mode: FakeTensorMode) -> None: ...
    def __torch_function__(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object] = ...,
        kwargs: Mapping[str, object] | None = ...,
    ) -> FakeTensor: ...

_DISPATCH_META_HANDLERS = ...
_DISPATCH_HANDLE_DIRECTLY = ...

def evict_fake_tensor_cache_key(key: _DispatchCacheKey) -> None: ...
@atexit.register
def dump_cache_stats() -> None: ...
def inferred_fake_kernel_from_real_out(
    mode: FakeTensorMode, op: torch._ops.OpOverload, real_out: Any
) -> Any: ...
