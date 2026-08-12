from collections.abc import Sequence
from types import NotImplementedType
from typing import Any, Self
from collections.abc import (
    Callable as Callable,
    Callable as _Callable,
)
from math import (
    e as e,
    inf as inf,
    nan as nan,
    pi as pi,
)
from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
    Any as _Any,
    TypeVar as _TypeVar,
    overload as _overload,
)
from typing_extensions import (
    ParamSpec as _ParamSpec,
    TypeIs as _TypeIs,
)

import builtins
import functools
import os
import platform
import sys

from torch import (
    _C as _C,
    _VF as _VF,
    Tensor as Tensor,
    __config__ as __config__,
    __future__ as __future__,
    _awaits as _awaits,
    _dynamo as _dynamo,
    _inductor as _inductor,
    _library as _library,
    _logging as _logging,
    _ops as _ops,
    _size_docs as _size_docs,
    _storage_docs as _storage_docs,
    _subclasses as _subclasses,
    _tensor_docs as _tensor_docs,
    _torch_docs as _torch_docs,
    accelerator as accelerator,
    amp as amp,
    ao as ao,
    autograd as autograd,
    backends as backends,
    compiler as compiler,
    cpu as cpu,
    cuda as cuda,
    distributed as distributed,
    distributions as distributions,
    export as export,
    fft as fft,
    func as func,
    functional as functional,
    futures as futures,
    fx as fx,
    hub as hub,
    jit as jit,
    library as library,
    linalg as linalg,
    masked as masked,
    mps as mps,
    mtia as mtia,
    multiprocessing as multiprocessing,
    nested as nested,
    nn as nn,
    onnx as onnx,
    optim as optim,
    overrides as overrides,
    profiler as profiler,
    quantization as quantization,
    quasirandom as quasirandom,
    random as random,
    return_types as return_types,
    serialization as serialization,
    sparse as sparse,
    special as special,
    storage as storage,
    testing as testing,
    types as types,
    utils as utils,
    version as version,
    xpu as xpu,
)
from torch._C import *
from torch._C import _initExtension as _initExtension
from torch._C._VariableFunctions import *
from torch._classes import classes as classes
from torch._compile import _disable_dynamo as _disable_dynamo
from torch._higher_order_ops import (
    cond as cond,
    while_loop as while_loop,
)
from torch._linalg_utils import (
    eig as eig,
    lstsq as lstsq,
    matrix_rank as matrix_rank,
    solve as solve,
)
from torch._lobpcg import lobpcg as lobpcg
from torch._ops import ops as ops
from torch._tensor import Tensor as Tensor
from torch._tensor_str import set_printoptions as set_printoptions
from torch._utils import (
    _import_dotted_name as _import_dotted_name,
    classproperty as classproperty,
)
from torch._utils_internal import (
    USE_GLOBAL_DEPS as USE_GLOBAL_DEPS,
    USE_RTLD_GLOBAL_WITH_LIBTORCH as USE_RTLD_GLOBAL_WITH_LIBTORCH,
    get_file_path as get_file_path,
    prepare_multiprocessing_environment as prepare_multiprocessing_environment,
    profiler_allow_cudagraph_cupti_lazy_reinit_cuda12 as profiler_allow_cudagraph_cupti_lazy_reinit_cuda12,
)
from torch.amp import (
    GradScaler as GradScaler,
    autocast as autocast,
)
from torch.autograd import (
    enable_grad as enable_grad,
    inference_mode as inference_mode,
    no_grad as no_grad,
    set_grad_enabled as set_grad_enabled,
)
from torch.func import vmap as vmap
from torch.functional import *
from torch.multiprocessing._atfork import register_after_fork as register_after_fork
from torch.random import (
    get_rng_state as get_rng_state,
    initial_seed as initial_seed,
    manual_seed as manual_seed,
    seed as seed,
    set_rng_state as set_rng_state,
)
from torch.serialization import (
    load as load,
    save as save,
)
from torch.signal import windows as windows
from torch.storage import (
    TypedStorage as TypedStorage,
    UntypedStorage as UntypedStorage,
    _LegacyStorage as _LegacyStorage,
    _StorageBase as _StorageBase,
    _warn_typed_storage_removal as _warn_typed_storage_removal,
)
from torch.torch_version import __version__ as __version__
from torch.types import (
    Device as Device,
    IntLikeType as IntLikeType,
)
from torch.utils.dlpack import (
    from_dlpack as from_dlpack,
    to_dlpack as to_dlpack,
)

import torch
import torch.fx.experimental.sym_node
import torch.nn.intrinsic
import torch.nn.qat
import torch.nn.quantizable
import torch.nn.quantized

from . import _rocm_init as _rocm_init

"""
The torch package contains data structures for multi-dimensional
tensors and defines mathematical operations over these tensors.
Additionally, it provides many utilities for efficient serialization of
Tensors and arbitrary types, and other useful utilities.

It has a CUDA counterpart, that enables you to run your tensor computations
on an NVIDIA GPU with compute capability >= 3.0.
"""
__all__ = [
    "BoolStorage",
    "BoolTensor",
    "ByteStorage",
    "ByteTensor",
    "CharStorage",
    "CharTensor",
    "DoubleStorage",
    "DoubleTensor",
    "FloatStorage",
    "FloatTensor",
    "GradScaler",
    "IntStorage",
    "IntTensor",
    "LongStorage",
    "LongTensor",
    "ShortStorage",
    "ShortTensor",
    "SymBool",
    "SymFloat",
    "SymInt",
    "Tensor",
    "TypedStorage",
    "UntypedStorage",
    "are_deterministic_algorithms_enabled",
    "autocast",
    "chunk",
    "compile",
    "cond",
    "e",
    "enable_grad",
    "export",
    "get_default_device",
    "get_deterministic_debug_mode",
    "get_device_module",
    "get_float32_matmul_precision",
    "get_rng_state",
    "inf",
    "inference_mode",
    "initial_seed",
    "is_deterministic_algorithms_warn_only_enabled",
    "is_storage",
    "is_tensor",
    "is_warn_always_enabled",
    "load",
    "lobpcg",
    "manual_seed",
    "matmul",
    "nan",
    "no_grad",
    "pi",
    "rand",
    "randn",
    "save",
    "seed",
    "set_default_device",
    "set_default_tensor_type",
    "set_deterministic_debug_mode",
    "set_float32_matmul_precision",
    "set_printoptions",
    "set_rng_state",
    "set_warn_always",
    "split",
    "stack",
    "sym_float",
    "sym_fresh_size",
    "sym_int",
    "sym_ite",
    "sym_max",
    "sym_min",
    "sym_not",
    "sym_sum",
    "typename",
    "unravel_index",
    "use_deterministic_algorithms",
    "vmap",
]
if sys.platform == "win32": ...
if (USE_RTLD_GLOBAL_WITH_LIBTORCH or os.getenv("TORCH_USE_RTLD_GLOBAL")) and (
    platform.system() != "Windows"
):
    old_flags = ...
else: ...

class SymInt:
    def __init__(self, node) -> None: ...
    def __bool__(self) -> bool: ...
    def __int__(self) -> int: ...
    def __index__(self): ...
    def __round__(self, ndigits=...) -> Self: ...
    def __truediv__(self, other) -> SymFloat | Any | NotImplementedType: ...
    def __rtruediv__(self, other) -> SymFloat | Any | NotImplementedType: ...
    def __floordiv__(self, other) -> SymFloat | float | NotImplementedType: ...
    def __rfloordiv__(self, other) -> SymFloat | float | NotImplementedType: ...
    def __pow__(self, other) -> NotImplementedType | SymFloat | Any | SymInt: ...
    def __rpow__(self, other) -> NotImplementedType | SymFloat | Any | SymInt: ...
    def __eq__(self, other: object) -> builtins.bool: ...
    def __lt__(self, other) -> builtins.bool: ...
    def __gt__(self, other) -> builtins.bool: ...
    def __le__(self, other) -> builtins.bool: ...
    def __ge__(self, other) -> builtins.bool: ...
    def __add__(self, other) -> SymInt: ...
    def __radd__(self, other) -> SymInt: ...
    def __rmul__(self, other) -> SymInt: ...
    def __mod__(self, other: IntLikeType) -> SymInt: ...
    def __mul__(self, other) -> SymInt: ...
    def __pow_by_natural__(self, other) -> SymInt: ...
    def __rpow_by_natural__(self, other) -> SymInt: ...
    def __int_truediv__(self, other) -> SymFloat: ...
    def __rint_truediv__(self, other) -> SymFloat: ...
    def __int_floordiv__(self, other) -> SymFloat: ...
    def __rint_floordiv__(self, other) -> SymFloat: ...
    def __sym_max__(self, other): ...
    def __sym_min__(self, other): ...
    def __sym_float__(self): ...
    def __neg__(self): ...
    def __sub__(self, other: IntLikeType) -> SymInt: ...
    def __rsub__(self, other: IntLikeType) -> SymInt: ...
    def __and__(self, other) -> SymInt: ...
    def __or__(self, other) -> SymInt: ...
    def __repr__(self): ...
    def __hash__(self) -> builtins.int: ...
    def as_integer_ratio(self) -> tuple[SymInt, builtins.int]: ...
    def bit_length(self) -> builtins.int: ...
    def conjugate(self) -> SymInt: ...

class SymFloat:
    def __init__(self, node) -> None: ...
    def __truediv__(self, other) -> NotImplementedType | SymFloat: ...
    def __rtruediv__(self, other) -> NotImplementedType | SymFloat: ...
    def __floordiv__(self, other) -> NotImplementedType | Any | SymFloat | float: ...
    def __rfloordiv__(self, other) -> NotImplementedType | Any | SymFloat | float: ...
    def __bool__(self): ...
    def __float__(self): ...
    def __pow__(self, other) -> NotImplementedType | SymFloat: ...
    def __rpow__(self, other) -> NotImplementedType | SymFloat: ...
    def __eq__(self, other: object) -> builtins.bool: ...
    def __lt__(self, other) -> builtins.bool: ...
    def __gt__(self, other) -> builtins.bool: ...
    def __le__(self, other) -> builtins.bool: ...
    def __ge__(self, other) -> builtins.bool: ...
    def __float_pow__(self, other) -> SymFloat: ...
    def __rfloat_pow__(self, other) -> SymFloat: ...
    def __float_truediv__(self, other) -> SymFloat: ...
    def __rfloat_truediv__(self, other) -> SymFloat: ...
    def __trunc__(self): ...
    def __sym_max__(self, other): ...
    def __sym_min__(self, other): ...
    def __sym_int__(self): ...
    def is_integer(self): ...
    def as_integer_ratio(self) -> tuple[builtins.int, builtins.int]: ...
    def __repr__(self): ...
    def __hash__(self) -> int: ...
    def conjugate(self) -> SymFloat: ...
    def hex(self) -> str: ...

class SymBool:
    def __init__(self, node) -> None: ...
    def __bool__(self): ...
    def __int__(self) -> int: ...
    def __and__(self, other) -> SymBool: ...
    def __or__(self, other) -> SymBool: ...
    def __sym_not__(self) -> SymBool: ...
    def __sym_ite__(self, then_val, else_val): ...
    def __eq__(self, other) -> builtins.bool: ...
    def __repr__(self): ...
    def __hash__(self) -> int: ...

def sym_not(a) -> bool: ...
def sym_float(a) -> SymFloat | float: ...
def sym_int(a) -> SymInt | int: ...
def sym_max(a, b) -> float: ...
def sym_min(a, b) -> float: ...
def sym_sum(args) -> int | float | bool | SymInt | SymFloat | SymBool: ...

sym_sqrt = ...

def sym_ite(b, t, f) -> Any: ...
def sym_fresh_size(expr) -> Number: ...
def typename(obj: _Any, /) -> str: ...
def is_tensor(obj: _Any, /) -> _TypeIs[torch.Tensor]: ...
def is_storage(obj: _Any, /) -> _TypeIs[TypedStorage | UntypedStorage]: ...

_GLOBAL_DEVICE_CONTEXT = ...

def get_default_device() -> torch.device: ...
def set_default_device(device: Device) -> None: ...
def set_default_tensor_type(t: type[torch.Tensor] | str, /) -> None: ...
def set_default_dtype(d: torch.dtype, /) -> None: ...
def use_deterministic_algorithms(
    mode: builtins.bool, *, warn_only: builtins.bool = ...
) -> None: ...
def are_deterministic_algorithms_enabled() -> builtins.bool: ...
def is_deterministic_algorithms_warn_only_enabled() -> builtins.bool: ...
def set_deterministic_debug_mode(debug_mode: builtins.int | str) -> None: ...
def get_deterministic_debug_mode() -> builtins.int: ...
def get_float32_matmul_precision() -> str: ...
def set_float32_matmul_precision(precision: str) -> None: ...
def set_warn_always(b: builtins.bool, /) -> None: ...
def is_warn_always_enabled() -> builtins.bool: ...

newaxis: None = ...

class ByteStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class DoubleStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class FloatStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class HalfStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class LongStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class IntStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class ShortStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class CharStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class BoolStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class BFloat16Storage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class ComplexDoubleStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class ComplexFloatStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class QUInt8Storage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class QInt8Storage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class QInt32Storage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class QUInt4x2Storage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class QUInt2x4Storage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

_storage_classes: set[type[TypedStorage | UntypedStorage]] = ...
_tensor_classes: set[type[torch.Tensor]] = ...
_segment_reduce = ...
PRIVATE_OPS = ...

def compiled_with_cxx11_abi() -> builtins.bool: ...

legacy_contiguous_format = ...
quantized_lstm = ...
quantized_gru = ...

class _TorchCompileInductorWrapper:
    compiler_name = ...
    def __init__(self, mode, options, dynamic) -> None: ...
    def __eq__(self, other) -> bool: ...
    def apply_mode(self, mode: str | None) -> None: ...
    def apply_options(self, options: dict[str, _Any] | None) -> None: ...
    def __call__(
        self, model_, inputs_
    ) -> Callable[[list[object]], Sequence[Tensor]] | str | list[str] | Weights: ...
    def get_compiler_config(self) -> dict[str, Any]: ...
    def reset(self) -> None: ...

class _TorchCompileWrapper:
    def __init__(self, backend, mode, options, dynamic) -> None: ...
    def __eq__(self, other) -> bool: ...
    def __call__(self, model_, inputs_) -> CompiledFn: ...
    def reset(self) -> None: ...

_InputT = _ParamSpec("_InputT")
_RetT = _TypeVar("_RetT")

_ModuleT = _TypeVar("_ModuleT", bound="torch.nn.Module")

# `compile` returns the module itself (an `OptimizedModule` proxying it), so the
# module type must survive; matching it against `Callable` cannot solve the
# `Module.__call__` self-type and erases the signature at every call site.
@_overload
def compile(
    model: _ModuleT,
    *,
    fullgraph: builtins.bool = ...,
    dynamic: builtins.bool | None = ...,
    backend: str | _Callable[..., _Any] = ...,
    mode: str | None = ...,
    options: dict[str, str | builtins.int | builtins.bool | _Callable[..., _Any]]
    | None = ...,
    disable: builtins.bool = ...,
) -> _ModuleT: ...
@_overload
def compile(
    model: _Callable[_InputT, _RetT],
    *,
    fullgraph: builtins.bool = ...,
    dynamic: builtins.bool | None = ...,
    backend: str | _Callable[..., _Any] = ...,
    mode: str | None = ...,
    options: dict[str, str | builtins.int | builtins.bool | _Callable[..., _Any]]
    | None = ...,
    disable: builtins.bool = ...,
) -> _Callable[_InputT, _RetT]: ...
@_overload
def compile(
    model: None = ...,
    *,
    fullgraph: builtins.bool = ...,
    dynamic: builtins.bool | None = ...,
    backend: str | _Callable[..., _Any] = ...,
    mode: str | None = ...,
    options: dict[str, str | builtins.int | builtins.bool | _Callable[..., _Any]]
    | None = ...,
    disable: builtins.bool = ...,
) -> _Callable[[_Callable[_InputT, _RetT]], _Callable[_InputT, _RetT]]: ...
def compile(
    model: _Callable[_InputT, _RetT] | None = ...,
    *,
    fullgraph: builtins.bool = ...,
    dynamic: builtins.bool | None = ...,
    backend: str | _Callable[..., _Any] = ...,
    mode: str | None = ...,
    options: dict[str, str | builtins.int | builtins.bool | _Callable[..., _Any]]
    | None = ...,
    disable: builtins.bool = ...,
) -> (
    _Callable[[_Callable[_InputT, _RetT]], _Callable[_InputT, _RetT]]
    | _Callable[_InputT, _RetT]
): ...

class _TritonLibrary:
    lib = ...
    ops_table: dict[tuple[str, str], _Callable] = ...
    @classmethod
    def registerOp(
        cls, op_key, full_schema, op_impl, dispatch_key
    ) -> Callable[..., Any]: ...

_deprecated_attrs = ...

@functools.cache
def get_device_module(device: torch.device | str | None = ...) -> Any: ...
def _check(
    cond: builtins.bool,
    message: str | Callable[[], str] | None = ...,
) -> None: ...

# Re-export class names that live in ``torch._C`` but were dropped from
# ``torch.__all__``. Stub-internal references like ``torch.memory_format``
# (used by ``Tensor.to(..., memory_format=...)``) and ``torch.Generator``
# (used by ``nn.init.*(..., generator=...)``) otherwise resolve to the
# ``torch`` module itself, producing ``Unknown`` in downstream call sites.
from torch._C import (  # noqa: E402 -- mirrors upstream module layout
    Generator as Generator,
    device as device,
    dtype as dtype,
    layout as layout,
    memory_format as memory_format,
)
