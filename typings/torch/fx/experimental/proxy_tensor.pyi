from collections.abc import Callable, Generator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Protocol,
    Self,
    TypeVar,
    Unpack,
    overload,
)
from typing_extensions import ParamSpec, TypeVarTuple

import threading
import types

from torch import Tensor, fx
from torch._library.fake_class_registry import FakeScriptObject
from torch._ops import OpOverload
from torch.fx import GraphModule, Proxy, Tracer
from torch.fx.node import Argument, Target
from torch.nn import Module
from torch.overrides import TorchFunctionMode
from torch.types import PySymType
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._stats import count
from torch.utils._thunk import Thunk

import sympy
import torch

from ._backward_state import BackwardState

__all__ = [
    "DecompositionInterpreter",
    "PythonKeyTracer",
    "dispatch_trace",
    "get_innermost_proxy_mode",
    "get_proxy_mode",
    "handle_sym_dispatch",
    "make_fx",
    "maybe_disable_thunkify",
    "maybe_enable_thunkify",
    "py_sym_types",
]
type _ProxyTracer = PythonKeyTracer | _GraphAppendingTracerEx
_AnyScriptObject = ...
type _AnyScriptObjectType = torch.ScriptObject | FakeScriptObject
aten = ...
prim = ...
log = ...
not_implemented_log = ...
CURRENT_DECOMPOSITION_TABLE: Mapping[OpOverload, Callable] = ...
CONSTANT_NUMEL_LIMIT = ...
T = TypeVar("T")
U = TypeVar("U")
_P = ParamSpec("_P")
R = TypeVar("R")
_Ts = TypeVarTuple("_Ts")
null_ctx_type = ...
_pytree_subclasses_that_lose_info = ...

def fake_signature(fn: Callable[_P, R], nargs: int) -> Callable[_P, R]: ...
@contextmanager
def decompose(
    decomposition_table: Mapping[OpOverload, Callable] | None,
) -> Generator[Mapping[OpOverload, Callable]]: ...

proxy_slot = ...

class _NoDefault: ...

no_default = ...

class _HasMeta(Protocol):
    meta: dict[str, PySymType]

def is_sym_node(node: _HasMeta) -> bool: ...
@overload
def set_proxy_slot(obj: Tensor, tracer: _ProxyTracer, proxy: _ProxyTensor) -> None: ...
@overload
def set_proxy_slot(
    obj: _AnyScriptObjectType, tracer: _ProxyTracer, proxy: Proxy
) -> None: ...
@overload
def set_proxy_slot(
    obj: PySymType, tracer: _ProxyTracer, proxy: _PySymProxyType
) -> None: ...

class _DisableUpdateTensorTracker(threading.local):
    value: bool = ...

_disable_update_tensor_tracker_tls = ...
_FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT: dict[int, torch.fx.Node] = ...

def set_proxy_slot(
    obj: PySymType | _AnyScriptObjectType | Tensor, tracer: _ProxyTracer, proxy: object
) -> None: ...
def has_proxy_slot(obj: Tensor, tracer: _ProxyTracer) -> bool: ...

type _PySymProxyType = Thunk[Proxy]

@overload
def get_proxy_slot(obj: Tensor, tracer: _ProxyTracer) -> _ProxyTensor: ...
@overload
def get_proxy_slot(
    obj: Tensor, tracer: _ProxyTracer, default: U
) -> _ProxyTensor | U: ...
@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[_ProxyTensor], R],
) -> R | U: ...
@overload
def get_proxy_slot(obj: _AnyScriptObjectType, tracer: _ProxyTracer) -> Proxy: ...
@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType, tracer: _ProxyTracer, default: U
) -> Proxy | U: ...
@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[Proxy], R],
) -> R | U: ...
@overload
def get_proxy_slot(obj: PySymType, tracer: _ProxyTracer) -> _PySymProxyType: ...
@overload
def get_proxy_slot(
    obj: PySymType, tracer: _ProxyTracer, default: T
) -> T | _PySymProxyType: ...
@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[_PySymProxyType], R],
) -> R | U: ...
def get_proxy_slot(
    obj: Tensor | _AnyScriptObjectType | PySymType,
    tracer: _ProxyTracer,
    default: object = ...,
    transform: Callable = ...,
) -> object: ...
def snapshot_fake(val: Tensor, include_real: bool = ...) -> Tensor | None: ...

type _ExtractValType = (
    PySymType
    | _AnyScriptObjectType
    | BackwardState
    | list[_ExtractValType]
    | tuple[_ExtractValType, ...]
    | dict[str, _ExtractValType]
    | Tensor
    | int
    | float
    | bool
    | None
)

def extract_val(val: _ExtractValType, include_real: bool = ...) -> _ExtractValType: ...
@contextmanager
def maybe_disable_thunkify() -> Generator[None]: ...
@contextmanager
def maybe_enable_thunkify() -> Generator[None]: ...
def set_meta(proxy: Proxy, val: _ExtractValType) -> Proxy: ...
def thunkify(
    tracer: _ProxyTracer, f: Callable[_P, R], *args: _P.args, **kwargs: _P.kwargs
) -> Thunk[R]: ...
def track_tensor(
    tensor: Tensor, proxy: Proxy, *, constant: Tensor | None, tracer: _ProxyTracer
) -> None: ...

type _NestedProxys = Proxy | Sequence[_NestedProxys] | Mapping[object, _NestedProxys]
type _NestedTensors = (
    Tensor | Sequence[_NestedTensors] | Mapping[object, _NestedTensors]
)

def track_tensor_tree(
    inner_res: T,
    proxy_res: _NestedProxys,
    *,
    constant: _NestedTensors | None,
    tracer: _ProxyTracer,
) -> T: ...

@dataclass
class _ProxyTensor:
    proxy: Proxy
    constant: Tensor | None

def fetch_sym_proxy(
    tracer: _ProxyTracer,
) -> Callable[[PySymType], bool | int | float | Proxy]: ...
@overload
def fetch_object_proxy(tracer: _ProxyTracer, t: Tensor) -> _ProxyTensor | Tensor: ...
@overload
def fetch_object_proxy(
    tracer: _ProxyTracer, t: _AnyScriptObjectType
) -> Proxy | _AnyScriptObjectType: ...
@overload
def fetch_object_proxy(
    tracer: _ProxyTracer, t: PySymType
) -> _PySymProxyType | PySymType: ...
def fetch_object_proxy(
    tracer: _ProxyTracer, t: Tensor | _AnyScriptObjectType | PySymType
) -> object: ...

HANDLED_TYPES = ...

def proxy_call(
    proxy_mode: ProxyTorchDispatchMode,
    func: OpOverload,
    pre_dispatch: bool,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object: ...

class _SymNodeDict:
    def __init__(self) -> None: ...
    def __setitem__(self, key: PySymType, value: _PySymProxyType) -> None: ...
    def __getitem__(self, key: PySymType) -> _PySymProxyType: ...
    def __contains__(self, key: PySymType) -> bool: ...
    def get(
        self, key: PySymType, default: _PySymProxyType | None = ...
    ) -> _PySymProxyType: ...
    def __iter__(self) -> Any: ...
    def __len__(self) -> int: ...

class PythonKeyTracer(Tracer):
    script_object_tracker: MutableMapping[_AnyScriptObjectType, Proxy]
    symnode_tracker: _SymNodeDict
    sympy_expr_tracker: dict[sympy.Symbol, object]
    tensor_tracker: MutableMapping[Tensor, _ProxyTensor]
    torch_fn_counts: dict[OpOverload, int]
    enable_thunkify: bool = ...
    def __init__(self) -> None: ...
    def call_module(
        self,
        m: Module,
        forward: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any: ...
    def getattr(
        self, attr: str, attr_val: object, parameter_proxy_cache: dict[str, Proxy]
    ) -> object: ...
    def create_arg(self, a: object) -> fx.node.Node: ...
    @overload
    def unwrap_proxy(self, e: Tensor) -> Proxy | Tensor: ...
    @overload
    def unwrap_proxy(self, e: PySymType) -> Proxy | PySymType: ...
    @overload
    def unwrap_proxy(self, e: _AnyScriptObjectType) -> Proxy | _AnyScriptObjectType: ...
    def unwrap_proxy(self, e: T) -> object: ...
    def create_node(
        self,
        kind: str,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        name: str | None = ...,
        type_expr: Any | None = ...,
    ) -> torch.fx.Node: ...

@torch._disable_dynamo
def dispatch_trace(
    root: Module | Callable, tracer: Tracer, concrete_args: tuple[Any, ...] | None = ...
) -> GraphModule: ...
def wrap_key(
    f: Callable[[Unpack[_Ts]], R],
    tensors: tuple[*_Ts],
    tracer: _ProxyTracer,
    pre_dispatch: bool,
) -> Callable[_P, R]: ...

ORIGINAL_ATEN: object | None = ...

@contextmanager
def set_original_aten_op(func: OpOverload) -> Generator[None]: ...

class TorchFunctionMetadataMode(TorchFunctionMode):
    def __init__(self, tracer: _ProxyTracer) -> None: ...
    def __torch_function__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = ...,
        kwargs: dict[str, object] | None = ...,
    ) -> object: ...

_temp_remove_metadata_torch_function_mode = ...

class PreDispatchTorchFunctionMode(TorchFunctionMode):
    def __init__(self, tracer: _ProxyTracer) -> None: ...
    def __torch_function__(
        self,
        func: OpOverload | Callable,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = ...,
        kwargs: dict[str, object] | None = ...,
    ) -> object: ...

_temp_remove_pre_dispatch_torch_function_mode = ...

class ProxyTorchDispatchMode(TorchDispatchMode):
    @property
    def enable_tracing(self) -> bool: ...
    def __init__(
        self,
        tracer: _ProxyTracer,
        tracing_mode: str,
        pre_dispatch: bool = ...,
        _allow_fake_constant: bool = ...,
        _error_on_data_dependent_ops: bool = ...,
    ) -> None: ...
    @count
    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = ...,
        kwargs: dict[str, object] | None = ...,
    ) -> object: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool | None: ...
    @classmethod
    def is_infra_mode(cls) -> bool: ...
    def __sym_dispatch__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object: ...

class _GraphAppendingTracerEx(fx.proxy.GraphAppendingTracer):
    script_object_tracker: MutableMapping[_AnyScriptObjectType, Proxy]
    symnode_tracker: MutableMapping[PySymType, _PySymProxyType]
    tensor_tracker: MutableMapping[Tensor, _ProxyTensor]
    sympy_expr_tracker: dict[sympy.Symbol, object]
    torch_fn_metadata: OpOverload | None
    torch_fn_counts: dict[OpOverload, int]
    enable_thunkify: bool = ...
    def __init__(self, graph: fx.graph.Graph) -> None: ...

class DecompositionInterpreter(fx.Interpreter):
    def __init__(
        self,
        module: fx.GraphModule,
        new_graph: fx.Graph,
        decomposition_table: Mapping[OpOverload, Callable] | None = ...,
        **kwargs: object,
    ) -> None: ...
    def placeholder(
        self, target: str, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> object: ...
    def get_attr(
        self, target: str, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> object: ...
    def output(
        self, target: str, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> object: ...
    def run(self, *args: object, **kwargs: object) -> object: ...

def wrapper_and_args_for_make_fx(
    func: Callable[..., R], args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[Callable[[list[object]], R], list[object]]: ...
@contextmanager
def disable_autocast_cache() -> Generator[None]: ...

class _ModuleNotInstalledAsSubmoduleError(NameError): ...

class _AttrProxy:
    def reset_proxy_mapping(self, base: Module, path: str) -> None: ...

class _ModuleStackTracer(PythonKeyTracer):
    def __init__(self, scope_root: GraphModule) -> None: ...
    def path_of_module(self, mod: Module) -> str: ...
    def getattr(
        self, attr: str, attr_val: object, parameter_proxy_cache: dict[str, Proxy]
    ) -> object: ...
    def trace(
        self, root: Module | Callable, concrete_args: dict[str, object] | None
    ) -> fx.Graph: ...
    def call_module(
        self,
        m: Module,
        forward: Callable,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None: ...
    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool: ...
    def create_node(self, *args: object, **kwargs: object) -> fx.node.Node: ...

class _MakefxTracer:
    def __init__(
        self,
        decomposition_table: Mapping[OpOverload, Callable] | None,
        tracing_mode: str,
        _allow_non_fake_inputs: bool,
        pre_dispatch: bool,
        record_module_stack: bool,
        _allow_fake_constant: bool,
        _error_on_data_dependent_ops: bool,
        record_stack_traces: bool = ...,
        parent_tracer: _MakefxTracer | None = ...,
    ) -> None: ...
    def trace(self, f: Callable, *args: object) -> fx.GraphModule: ...
    def is_hop_subgraph_tracer(self) -> bool: ...
    def trace_subgraph(self, f: Callable, *args: object) -> GraphModule: ...

_CURRENT_MAKE_FX_TRACER: _MakefxTracer | None = ...

def make_fx(
    f: Callable,
    decomposition_table: Mapping[OpOverload, Callable] | None = ...,
    tracing_mode: str = ...,
    _allow_non_fake_inputs: bool = ...,
    *,
    pre_dispatch: bool = ...,
    record_module_stack: bool = ...,
    _allow_fake_constant: bool = ...,
    _error_on_data_dependent_ops: bool = ...,
    record_stack_traces: bool = ...,
) -> Callable[..., GraphModule]: ...
def get_torch_dispatch_modes() -> list[TorchDispatchMode]: ...
def get_innermost_proxy_mode() -> ProxyTorchDispatchMode | None: ...
def get_proxy_mode() -> ProxyTorchDispatchMode | None: ...
def handle_sym_dispatch(
    func: Callable[_P, R], args: _P.args, kwargs: _P.kwargs
) -> R: ...
@contextmanager
def disable_proxy_modes_tracing() -> Generator[ProxyTorchDispatchMode]: ...
def maybe_handle_decomp(
    proxy_mode: ProxyTorchDispatchMode,
    op: OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object: ...
def get_isolated_graphmodule(
    func: Callable,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    tracing_mode: str = ...,
    decomposition_table: Mapping[OpOverload, Callable] | None = ...,
) -> GraphModule: ...
