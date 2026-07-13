from collections.abc import Callable
from typing import TypeVar
from typing_extensions import ParamSpec

"""Internal flags for ONNX export."""
_is_onnx_exporting = ...
_P = ParamSpec("_P")
_R = TypeVar("_R")

def set_onnx_exporting_flag(func: Callable[_P, _R]) -> Callable[_P, _R]: ...
