from collections.abc import Generator
from typing import Any
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar
from typing_extensions import ParamSpec

from torch._C import DispatchKey

import torch
import torch._C
import torch._ops

__all__ = ["enable_pre_dispatch", "enable_python_dispatcher", "no_python_dispatcher"]
no_python_dispatcher = torch._C._DisablePythonDispatcher
enable_python_dispatcher = torch._C._EnablePythonDispatcher
enable_pre_dispatch = torch._C._EnablePreDispatch
CROSSREF_FUNCTIONALIZE = ...
_P = ParamSpec("_P")
_T = TypeVar("_T")

def all_py_loaded_overloads() -> Iterator[torch._ops.OpOverload]: ...
@contextmanager
def suspend_functionalization() -> Generator[None, Any, None]: ...
def check_tensor_metadata_matches(nv, rv, desc) -> None: ...
def check_metadata_matches(n, r, desc) -> None: ...

class Lit:
    def __init__(self, s) -> None: ...
    def __repr__(self) -> Any: ...

def make_crossref_functionalize(
    op: torch._ops.OpOverload[_P, _T], final_key: DispatchKey
) -> Callable[_P, _T] | DispatchKey: ...
@contextmanager
def enable_crossref_functionalize() -> Generator[None, Any, None]: ...
