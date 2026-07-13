import abc
import typing as t

from torch.fx._compatibility import compatibility

import torch
import torch.fx

__all__ = [
    "OpSupports",
    "OperatorSupport",
    "OperatorSupportBase",
    "any_chain",
    "chain",
    "create_op_support",
]
TargetTypeName = str
type SupportedArgumentDTypes = (
    tuple[t.Sequence[t.Sequence[torch.dtype]], dict[str, t.Sequence[torch.dtype]]]
    | None
)
type SupportDict = t.Mapping[TargetTypeName, SupportedArgumentDTypes]

@compatibility(is_backward_compatible=False)
class OperatorSupportBase(abc.ABC):
    @abc.abstractmethod
    def is_node_supported(
        self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool: ...

@compatibility(is_backward_compatible=False)
class OperatorSupport(OperatorSupportBase):
    _support_dict: SupportDict
    def __init__(self, support_dict: SupportDict | None = ...) -> None: ...
    def is_node_supported(
        self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool: ...

type IsNodeSupported = t.Callable[
    [t.Mapping[str, torch.nn.Module], torch.fx.Node], bool
]

@compatibility(is_backward_compatible=False)
def create_op_support(is_node_supported: IsNodeSupported) -> OperatorSupportBase: ...
@compatibility(is_backward_compatible=False)
def chain(*op_support: OperatorSupportBase) -> OperatorSupportBase: ...
@compatibility(is_backward_compatible=False)
def any_chain(*op_support: OperatorSupportBase) -> OperatorSupportBase: ...

@compatibility(is_backward_compatible=False)
class OpSupports:
    @classmethod
    def decline_if_input_dtype(cls, dtype: torch.dtype) -> OperatorSupportBase: ...
    @classmethod
    def decline_if_node_in_names(
        cls, disallow_set: set[str]
    ) -> OperatorSupportBase: ...
