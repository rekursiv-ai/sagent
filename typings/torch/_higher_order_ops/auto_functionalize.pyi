from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from torch import Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.utils import HopInstance
from torch._ops import HigherOrderOperator, OperatorBase, OpOverload
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

import torch
import torch.utils._pytree as pytree

class SchemaHolder:
    def __init__(self, schema: torch.FunctionSchema) -> None: ...
    def __eq__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
    @classmethod
    def from_tree_spec(cls, tree_spec: pytree.TreeSpec) -> Self: ...

def get_base(tensor): ...

class ViewInfo(ABC):
    base_index: int
    def __init__(self, base_index) -> None: ...
    @abstractmethod
    def regenerate_view(self, bases_list: list[Tensor]) -> None: ...

@dataclass
class AsStridedViewInfo(ViewInfo):
    size: Sequence[int | torch.SymInt]
    stride: Sequence[int | torch.SymInt]
    storage_offset: int
    def __init__(self, base_index, size, stride, storage_offset) -> None: ...
    def regenerate_view(self, bases_list: list[Tensor]) -> Tensor: ...

@dataclass
class SliceViewInfo(ViewInfo):
    dim: int | torch.SymInt
    start: int | torch.SymInt
    end: int | torch.SymInt
    def __init__(self, base_index, dim, start, end) -> None: ...
    def regenerate_view(self, bases_list: list[Tensor]) -> Any: ...

@dataclass
class AliasViewInfo(ViewInfo):
    def __init__(self, base_index) -> None: ...
    def regenerate_view(self, bases_list: list[Tensor]) -> Any: ...

@dataclass
class NotView(ViewInfo):
    def __init__(self, base_index) -> None: ...
    def regenerate_view(self, bases_list: list[Tensor]) -> Tensor: ...

def is_alias(base, tensor) -> bool: ...
def try_use_slice(
    base, tensor
) -> tuple[Literal[0], Literal[0], Any] | tuple[int | None, Any, Any] | None: ...
def write_view_information_to_args(
    mutable_arg_names: list[str],
    mutable_arg_types: list[torch.Type],
    kwargs: dict[str, Any],
    arg_to_base_index: dict[str, Any],
) -> None: ...
def read_view_information_from_args(
    mutable_arg_names: list[str],
    mutable_arg_types: list[torch.Type],
    kwargs: dict[str, Any],
    all_bases: list[Tensor],
) -> dict[str, Any]: ...

class AutoFunctionalized(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(
        self, /, _mutable_op: OpOverload, **kwargs: Any
    ) -> tuple[Any, tuple[Tensor, ...]]: ...

auto_functionalized = ...
type _MutableOpType = OpOverload | HigherOrderOperator

class AutoFunctionalizedV2(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(
        self, /, _mutable_op: _MutableOpType, **kwargs: Any
    ) -> tuple[Any, tuple[Tensor, ...]]: ...

auto_functionalized_v2 = ...

def can_auto_functionalize(op: OperatorBase | HopInstance) -> bool: ...
def get_mutable_args_from_schema(
    schema: torch.FunctionSchema,
) -> tuple[list[str], list[torch.Type]]: ...
def get_mutable_args(op: OpOverload) -> tuple[list[str], list[torch.Type]]: ...
def do_auto_functionalize(
    mode: torch._subclasses.functional_tensor.FunctionalTensorMode,
    op: OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any: ...

class FunctionalCallableWithEpilogue:
    def __init__(self, orig_callable: Callable) -> None: ...
    def __call__(self, *args, **kwargs) -> tuple[Any, ...]: ...
    def __hash__(self) -> int: ...

def do_auto_functionalize_v2(
    mode: torch._subclasses.functional_tensor.FunctionalTensorMode,
    op: OpOverload | HopInstance,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any: ...
@auto_functionalized.py_impl(DispatchKey.CompositeExplicitAutograd)
def auto_functionalized_dense(
    _mutable_op: OpOverload,
    _only_clone_these_tensors: tuple[str, ...] | None = ...,
    **kwargs: Any,
) -> tuple[Any, tuple[Tensor, ...]]: ...
@auto_functionalized.py_impl(FakeTensorMode)
def auto_functionalized_fake(
    mode, _mutable_op: OpOverload, **kwargs: Any
) -> tuple[Any, tuple[Tensor, ...]]: ...
@auto_functionalized.py_impl(ProxyTorchDispatchMode)
def auto_functionalized_proxy(
    mode, _mutable_op: OpOverload, **kwargs: Any
) -> tuple[Any, tuple[Tensor, ...]]: ...
@auto_functionalized.py_functionalize_impl
def auto_functionalized_func(ctx, _mutable_op, **kwargs): ...
@auto_functionalized_v2.py_impl(DispatchKey.CompositeExplicitAutograd)
def auto_functionalized_v2_dense(
    _mutable_op: _MutableOpType,
    _only_clone_these_bases: tuple[int, ...] | None = ...,
    **kwargs: Any,
) -> tuple[Any, tuple[Tensor, ...]]: ...
@auto_functionalized_v2.py_impl(FakeTensorMode)
def auto_functionalized_v2_fake(
    mode, _mutable_op: _MutableOpType, **kwargs: dict[str, Any]
) -> tuple[Any, tuple[Tensor, ...]]: ...
@auto_functionalized_v2.py_impl(ProxyTorchDispatchMode)
def auto_functionalized_v2_proxy(
    mode, _mutable_op: _MutableOpType, **kwargs: Any
) -> tuple[Any, tuple[Tensor, ...]]: ...
@auto_functionalized_v2.py_functionalize_impl
def auto_functionalized_v2_func(ctx, _mutable_op, **kwargs): ...
