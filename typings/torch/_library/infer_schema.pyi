from typing import Any

import inspect
import typing

from torch.utils._exposed_in import exposed_in

import torch

_TestTensor = torch.Tensor

@exposed_in("torch.library")
def infer_schema(
    prototype_function: typing.Callable, /, *, mutates_args, op_name: str | None = ...
) -> str: ...
def derived_types(
    base_type: type | typing._SpecialForm,
    cpp_type: str,
    list_base: bool,
    optional_base_list: bool,
    optional_list_base: bool,
) -> list[tuple[type | _SpecialForm | GenericAlias, str]]: ...
def get_supported_param_types() -> dict[Any, Any]: ...

SUPPORTED_RETURN_TYPES = ...

def parse_return(annotation, error_fn) -> str: ...

SUPPORTED_PARAM_TYPES = ...

def supported_param(param: inspect.Parameter) -> bool: ...
def tuple_to_list(tuple_type: type[tuple]) -> type[list]: ...
