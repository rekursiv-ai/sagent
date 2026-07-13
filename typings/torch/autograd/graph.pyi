from collections.abc import Callable, Generator, Iterable, Sequence
from typing import Any, Literal, NamedTuple

import abc
import contextlib

from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle

import torch

__all__ = [
    "GradientEdge",
    "Node",
    "allow_mutation_on_saved_tensors",
    "disable_saved_tensors_hooks",
    "get_gradient_edge",
    "increment_version",
    "register_multi_grad_hook",
    "save_on_cpu",
    "saved_tensors_hooks",
]
log = ...

class Node(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str: ...
    @property
    @abc.abstractmethod
    def next_functions(self) -> tuple[tuple[Node | None, int], ...]: ...
    @abc.abstractmethod
    def metadata(self) -> dict: ...
    @abc.abstractmethod
    def register_hook(self, fn: Callable[..., Any]) -> RemovableHandle: ...
    @abc.abstractmethod
    def register_prehook(self, fn: Callable[..., Any]) -> RemovableHandle: ...
    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool: ...

class GradientEdge(NamedTuple):
    node: Node
    output_nr: int
    ownership_token: Node | None = ...

def get_gradient_edge(tensor: torch.Tensor) -> GradientEdge: ...
def increment_version(tensor: torch.Tensor | Iterable[torch.Tensor]) -> None: ...

class saved_tensors_hooks:
    def __init__(
        self,
        pack_hook: Callable[[torch.Tensor], Any],
        unpack_hook: Callable[[Any], torch.Tensor],
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: object) -> None: ...

class save_on_cpu(saved_tensors_hooks):
    def __init__(self, pin_memory: bool = ..., device_type: str = ...) -> None: ...

@contextlib.contextmanager
def disable_saved_tensors_hooks(error_message: str) -> Generator[None]: ...

class _MultiHandle(RemovableHandle):
    handles: tuple[RemovableHandle, ...]
    def __init__(self, handles: tuple[RemovableHandle, ...]) -> None: ...
    def remove(self) -> None: ...
    def __getstate__(self) -> tuple[RemovableHandle, ...]: ...
    def __setstate__(self, state: tuple[RemovableHandle, ...]) -> None: ...

def register_multi_grad_hook(
    tensors: Sequence[torch.Tensor],
    fn: Callable[[Sequence[torch.Tensor | None]], None]
    | Callable[[torch.Tensor], None],
    *,
    mode: Literal["all", "any"] = ...,
) -> RemovableHandle: ...

_allow_mutation_on_saved_tensors_enabled: bool = ...
type _TID = tuple[int, int, int]
type _SID = tuple[int, int]

class _Handle: ...

class _swap_with_cloned(saved_tensors_hooks):
    def __init__(self, ctx: _AllowMutationOnSavedContext) -> None: ...

class _CloneArgBeforeMutateMode(TorchDispatchMode):
    def __init__(self, ctx: _AllowMutationOnSavedContext) -> None: ...
    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: Iterable[type],
        args: tuple[Any, ...] = ...,
        kwargs: dict[Any, Any] | None = ...,
    ) -> Any: ...

class _AllowMutationOnSavedContext:
    def __init__(self) -> None: ...
    def clear(self) -> None: ...

@contextlib.contextmanager
def allow_mutation_on_saved_tensors() -> Generator[_AllowMutationOnSavedContext]: ...
