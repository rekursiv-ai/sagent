from dataclasses import dataclass
from typing import Any

from torch.fx.node import Target

import torch
import torch.utils._pytree as pytree

@dataclass(frozen=True)
class HopArgumentInfo:
    name: str
    example_value: Any
    default_value: Any
    is_mutated: bool
    kw_only: bool

class HopArgumentInfoGen:
    @staticmethod
    def from_example(
        example_value: Any,
        *,
        name: str = ...,
        default_value: Any | None = ...,
        is_mutated: bool = ...,
        kw_only: bool = ...,
    ) -> HopArgumentInfo: ...

class CTypeGen:
    convert_to_base_ty = ...
    @staticmethod
    def from_example(obj: Any) -> Any: ...

class CArgumentGen:
    @staticmethod
    def from_hop_argument_info(
        arg_idx: int, arg_info: HopArgumentInfo, is_output: bool = ...
    ) -> Any: ...

class HopSchemaGenerator:
    def __init__(self, hop: torch._ops.HigherOrderOperator) -> None: ...
    def add_arg(
        self,
        name: str,
        example_value: Any,
        default_value: Any | None = ...,
        is_mutated: bool = ...,
        kw_only: bool = ...,
    ) -> None: ...
    def add_output(self, output: Any) -> None: ...
    def add_schema_tree_spec(self, *args: Any, **kwargs: Any) -> None: ...
    def gen_schema(self) -> torch._C.FunctionSchema: ...

class CFunctionSchemaGen:
    @staticmethod
    def from_hop_argument_info(
        op_name: str,
        inp_argument_info: list[HopArgumentInfo],
        out_argument_info: HopArgumentInfo,
        schema_tree_spec: pytree.TreeSpec | None,
    ) -> Any: ...

class HopSchema(torch._C.FunctionSchema):
    def __init__(
        self,
        name: str,
        overload_name: str,
        arguments: list[torch._C.Argument],
        returns: list[torch._C.Argument],
        is_vararg: bool,
        is_varret: bool,
        schema_tree_spec: pytree.TreeSpec | None,
    ) -> None: ...
    def __deepcopy__(self, memo: Any) -> HopSchema: ...

def find_hop_schema(
    gm: torch.fx.GraphModule, target: Target
) -> list[torch._C.FunctionSchema]: ...
