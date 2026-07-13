from collections.abc import Callable
from typing import TypeVar
from typing_extensions import ParamSpec

from torch.onnx._internal.exporter import _registration

"""Registry for aten functions."""
__all__ = ["get_torchlib_ops", "onnx_impl"]
_P = ParamSpec("_P")
_R = TypeVar("_R")
logger = ...
_registry: list[_registration.OnnxDecompMeta] = ...

def onnx_impl(
    target: _registration.TorchOp | tuple[_registration.TorchOp, ...],
    *,
    trace_only: bool = ...,
    complex: bool = ...,
    opset_introduced: int = ...,
    no_compile: bool = ...,
    private: bool = ...,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...
def get_torchlib_ops() -> tuple[_registration.OnnxDecompMeta, ...]: ...
