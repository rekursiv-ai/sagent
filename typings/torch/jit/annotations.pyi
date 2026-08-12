from torch import Tensor
from types import FunctionType, MethodType
from typing import Any, TypeIs
import enum

class Module:
    def __init__(self, name, members) -> None: ...
    def __getattr__(self, name): ...

class EvalEnv:
    env = ...
    def __init__(self, rcb) -> None: ...
    def __getitem__(
        self, name
    ) -> (
        type[
            Tensor
            | Tuple[Any, ...]
            | List[Any]
            | Dict[Any, Any]
            | Optional
            | Union
            | Future[Any]
            | _Await[Any]
        ]
        | Module
        | Any
        | None
    ): ...

def get_signature(fn, rcb, loc, is_method) -> tuple[list[Any], Any] | None: ...
def is_function_or_method(
    the_callable,
) -> TypeIs[FunctionType] | TypeIs[MethodType]: ...
def is_vararg(the_callable) -> bool: ...
def get_param_names(fn, n_args) -> list[str]: ...
def check_fn(fn, loc) -> None: ...
def parse_type_line(type_line, rcb, loc) -> tuple[list[Any], Any]: ...
def get_type_line(source) -> None: ...
def split_type_line(type_line) -> tuple[Any, Any]: ...
def try_real_annotations(fn, loc) -> tuple[list[Any], Any] | None: ...
def get_enum_value_type(e: type[enum.Enum], loc) -> AnyType | JitType: ...
def is_tensor(ann) -> bool: ...
def try_ann_to_type(ann, loc, rcb=...): ...
def ann_to_type(ann, loc, rcb=...): ...

__all__ = [
    "Any",
    "AnyType",
    "BroadcastingList1",
    "BroadcastingList2",
    "BroadcastingList3",
    "ComplexType",
    "Dict",
    "DictType",
    "FloatType",
    "IntType",
    "List",
    "ListType",
    "Module",
    "StringType",
    "TensorType",
    "Tuple",
    "TupleType",
    "ann_to_type",
    "check_fn",
    "get_param_names",
    "get_signature",
    "get_type_line",
    "is_dict",
    "is_list",
    "is_optional",
    "is_tuple",
    "is_union",
    "parse_type_line",
    "split_type_line",
    "try_ann_to_type",
    "try_real_annotations",
]
