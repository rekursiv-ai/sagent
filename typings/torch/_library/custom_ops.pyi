from collections.abc import Generator
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from typing import Any, Generic, Protocol, TypeVar, overload
from typing_extensions import ParamSpec

import weakref

from torch import _C
from torch.types import _dtype
from torch.utils._exposed_in import exposed_in

import torch

type device_types_t = str | Sequence[str] | None
_F = TypeVar("_F", bound=Callable[..., object])
_P = ParamSpec("_P")
_R = TypeVar("_R")
log = ...

class _CustomOpDecorator(Protocol):
    # Generic __call__, not a generic alias: the typevars must be solved where
    # the decorator is applied, not at the custom_op() call that returns it.
    def __call__[**P, R](self, fn: Callable[P, R], /) -> CustomOpDef[P, R]: ...

@overload
def custom_op(
    name: str,
    fn: None = ...,
    /,
    *,
    mutates_args: str | Iterable[str],
    device_types: device_types_t = ...,
    schema: str | None = ...,
) -> _CustomOpDecorator: ...
@overload
def custom_op(
    name: str,
    fn: Callable[_P, _R],
    /,
    *,
    mutates_args: str | Iterable[str],
    device_types: device_types_t = ...,
    schema: str | None = ...,
) -> CustomOpDef[_P, _R]: ...
@exposed_in("torch.library")
def custom_op(
    name: str,
    fn: Callable[_P, _R] | None = ...,
    /,
    *,
    mutates_args: str | Iterable[str],
    device_types: device_types_t = ...,
    schema: str | None = ...,
    tags: Sequence[_C.Tag] | None = ...,
) -> _CustomOpDecorator | CustomOpDef[_P, _R]: ...

class CustomOpDef(Generic[_P, _R]):
    def __init__(
        self,
        namespace: str,
        name: str,
        schema: str,
        fn: Callable,
        tags: Sequence[_C.Tag] | None = ...,
    ) -> None: ...
    @contextmanager
    def set_kernel_enabled(
        self, device_type: str, enabled: bool = ...
    ) -> Generator[None, Any, None]: ...
    @overload
    def register_kernel(self, device_types: device_types_t, fn: _F, /) -> _F: ...
    @overload
    def register_kernel(
        self, device_types: device_types_t, fn: None = ..., /
    ) -> Callable[[_F], _F]: ...
    def register_fake(self, fn: _F, /) -> _F: ...
    def register_torch_dispatch(
        self, torch_dispatch_class: Any, fn: Callable | None = ..., /
    ) -> Callable: ...
    def register_autograd(
        self,
        backward: Callable[..., object],
        /,
        *,
        setup_context: Callable[..., object] | None = ...,
    ) -> None: ...
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R: ...
    def register_vmap(
        self, func: Callable | None = ...
    ) -> Callable[..., None] | None: ...
    def register_autocast(
        self, device_type: str, cast_inputs: _dtype
    ) -> Callable[..., Any]: ...

def increment_version(val: Any) -> None: ...

OPDEF_TO_LIB: dict[str, torch.library.Library] = ...
OPDEFS: weakref.WeakValueDictionary = ...

def get_library_allowing_overwrite(
    namespace: str, name: str
) -> torch.library.Library: ...
