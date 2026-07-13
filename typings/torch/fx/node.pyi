from collections.abc import Callable, Mapping, Sequence
from typing import Any
from typing_extensions import ParamSpec, TypeVar

from torch._C import _NodeBase
from torch.fx.operator_schemas import ArgsKwargsPair

import torch

from ._compatibility import compatibility
from .graph import Graph

__all__ = ["Node", "has_side_effect", "map_aggregate", "map_arg"]
log = ...
type BaseArgumentTypes = (
    str
    | int
    | float
    | bool
    | complex
    | torch.dtype
    | torch.Tensor
    | torch.device
    | torch.memory_format
    | torch.layout
    | torch._ops.OpOverload
    | torch.SymInt
    | torch.SymBool
    | torch.SymFloat
)
base_types = ...
type Target = Callable[..., Any] | str
type Argument = (
    tuple[Argument, ...]
    | Sequence[Argument]
    | Mapping[str, Argument]
    | slice
    | range
    | Node
    | BaseArgumentTypes
    | None
)
ArgumentT = TypeVar("ArgumentT", bound=Argument)
_P = ParamSpec("_P")
_R = TypeVar("_R")
_legal_ops = ...
_side_effectful_need_to_be_preserved_pre_dispatch: list[Callable[..., Any]] = ...
_side_effectful_functions: set[Callable[..., Any]] = ...

@compatibility(is_backward_compatible=False)
def has_side_effect(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...

@compatibility(is_backward_compatible=True)
class Node(_NodeBase):
    _args: tuple[Argument, ...]
    _kwargs: dict[str, Argument]
    graph: Graph
    name: str
    op: str
    target: Target
    _input_nodes: dict[Node, None]
    users: dict[Node, None]
    type: Any | None
    _sort_key: Any
    _repr_fn: Callable[[Node], str] | None
    meta: dict[str, Any]
    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        graph: Graph,
        name: str,
        op: str,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        return_type: Any | None = ...,
    ) -> None: ...
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    @property
    def next(self) -> Node: ...
    @property
    def prev(self) -> Node: ...
    @compatibility(is_backward_compatible=True)
    def prepend(self, x: Node) -> None: ...
    def __gt__(self, other: Node) -> bool: ...
    def __lt__(self, other: Node) -> bool: ...
    def __ge__(self, other: Node) -> bool: ...
    def __le__(self, other: Node) -> bool: ...
    @compatibility(is_backward_compatible=True)
    def append(self, x: Node) -> None: ...
    @property
    def args(self) -> tuple[Argument, ...]: ...
    @args.setter
    def args(self, a: tuple[Argument, ...]) -> None: ...
    @property
    def kwargs(self) -> dict[str, Argument]: ...
    @kwargs.setter
    def kwargs(self, k: dict[str, Argument]) -> None: ...
    @property
    def all_input_nodes(self) -> list[Node]: ...
    @compatibility(is_backward_compatible=True)
    def update_arg(self, idx: int, arg: Argument) -> None: ...
    @compatibility(is_backward_compatible=True)
    def insert_arg(self, idx: int, arg: Argument) -> None: ...
    @compatibility(is_backward_compatible=True)
    def update_kwarg(self, key: str, arg: Argument) -> None: ...
    @property
    def stack_trace(self) -> str | None: ...
    @stack_trace.setter
    def stack_trace(self, trace: str | None) -> None: ...
    @compatibility(is_backward_compatible=True)
    def format_node(
        self,
        placeholder_names: list[str] | None = ...,
        maybe_return_typename: list[str] | None = ...,
        *,
        include_tensor_metadata: bool = ...,
    ) -> str | None: ...
    @compatibility(is_backward_compatible=True)
    def replace_all_uses_with(
        self,
        replace_with: Node,
        delete_user_cb: Callable[[Node], bool] = ...,
        *,
        propagate_meta: bool = ...,
    ) -> list[Node]: ...
    @compatibility(is_backward_compatible=False)
    def is_impure(self, impure_random: bool = ...) -> bool: ...
    @compatibility(is_backward_compatible=False)
    def normalized_arguments(
        self,
        root: torch.nn.Module,
        arg_types: tuple[Any] | None = ...,
        kwarg_types: dict[str, Any] | None = ...,
        normalize_to_only_use_kwargs: bool = ...,
    ) -> ArgsKwargsPair | None: ...
    @compatibility(is_backward_compatible=True)
    def replace_input_with(self, old_input: Node, new_input: Node) -> None: ...
    def __setattr__(self, name: str, value: Any) -> None: ...

@compatibility(is_backward_compatible=True)
def map_arg(a: ArgumentT, fn: Callable[[Node], Argument]) -> ArgumentT: ...
@compatibility(is_backward_compatible=True)
def map_aggregate(a: ArgumentT, fn: Callable[[Argument], Argument]) -> ArgumentT: ...
