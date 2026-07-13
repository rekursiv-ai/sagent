from collections.abc import Callable, Sequence
from typing import (
    Any,
    TypeVar,
    overload,
)
from typing_extensions import ParamSpec, deprecated

import functools

from torch._library.custom_ops import CustomOpDef, device_types_t
from torch.types import _dtype

import torch

__all__ = [
    "Library",
    "custom_op",
    "define",
    "fallthrough_kernel",
    "get_ctx",
    "get_kernel",
    "impl",
    "impl_abstract",
    "infer_schema",
    "register_autocast",
    "register_fake",
    "register_torch_dispatch",
    "register_vmap",
    "triton_op",
    "wrap_triton",
]
_T = TypeVar("_T")
_P = ParamSpec("_P")
_impls: set[str] = ...
_defs: set[str] = ...
_reserved_namespaces = ...

def fallthrough_kernel(): ...

class Library:
    def __init__(self, ns, kind, dispatch_key=...) -> None: ...
    def define(self, schema, alias_analysis=..., *, tags=...) -> Any: ...
    def impl(
        self, op_name, fn, dispatch_key=..., *, with_keyset=..., allow_override=...
    ) -> None: ...
    def fallback(self, fn, dispatch_key=..., *, with_keyset=...) -> None: ...

_keep_alive: list[Library] = ...
NAMELESS_SCHEMA = ...

@functools.singledispatch
def define(qualname, schema, *, lib=..., tags=...) -> None: ...
@define.register
def _(lib: Library, schema, alias_analysis=...) -> Callable[..., Any]: ...
@overload
def impl(
    qualname: str,
    types: str | Sequence[str],
    func: None = ...,
    *,
    lib: Library | None = ...,
) -> Callable[[Callable[..., object]], None]: ...
@overload
def impl(
    qualname: str,
    types: str | Sequence[str],
    func: Callable[..., object],
    *,
    lib: Library | None = ...,
) -> None: ...
@overload
def impl(
    lib: Library, name: str, dispatch_key: str = ...
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
@functools.singledispatch
def impl(
    qualname: str,
    types: str | Sequence[str],
    func: Callable[_P, _T] | None = ...,
    *,
    lib: Library | None = ...,
) -> object: ...
@deprecated(
    "`torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.",
    category=FutureWarning,
)
def impl_abstract(
    qualname, func=..., *, lib=..., _stacklevel=...
) -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Any]: ...

type _op_identifier = (
    str | torch._ops.OpOverload | torch._library.custom_ops.CustomOpDef
)

def register_kernel(
    op: _op_identifier,
    device_types: device_types_t,
    func: Callable | None = ...,
    /,
    *,
    lib: Library | None = ...,
) -> Callable[..., Any] | Callable[[Callable[..., object]], None] | None: ...
def register_autocast(
    op: _op_identifier,
    device_type: str,
    cast_inputs: _dtype,
    /,
    *,
    lib: Library | None = ...,
) -> Callable[..., Any] | None: ...
def register_fake(
    op: _op_identifier,
    func: Callable | None = ...,
    /,
    *,
    lib: Library | None = ...,
    _stacklevel: int = ...,
    allow_override: bool = ...,
) -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Any]: ...
def register_autograd(
    op: _op_identifier,
    backward: Callable,
    /,
    *,
    setup_context: Callable | None = ...,
    lib=...,
) -> None: ...
def register_torch_dispatch(
    op: _op_identifier,
    torch_dispatch_class: Any,
    func: Callable | None = ...,
    /,
    *,
    lib: Library | None = ...,
) -> Callable[..., Any]: ...
def register_vmap(
    op: _op_identifier, func: Callable | None = ..., /, *, lib=...
) -> Callable[..., None] | None: ...
def get_ctx() -> torch._library.fake_impl.FakeImplCtx: ...
def get_kernel(
    op: _op_identifier, dispatch_key: str | torch.DispatchKey
) -> torch._C._SafeKernelFunction: ...

_OPCHECK_DEFAULT_UTILS = ...

def opcheck(
    op: torch._ops.OpOverload | torch._ops.OpOverloadPacket | CustomOpDef,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = ...,
    *,
    test_utils: str | Sequence[str] = ...,
    raise_exception: bool = ...,
    atol=...,
    rtol=...,
) -> dict[str, str]: ...
