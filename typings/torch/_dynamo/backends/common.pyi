from collections.abc import Callable, Iterable
from typing import Any
from typing_extensions import ParamSpec, TypeVar

import torch

"""
This module provides common utilities and base classes for TorchDynamo backends.

Key components:
- AotAutograd: Base class for implementing AOT (Ahead-of-Time) autograd backends
- Backend utilities for handling:
  - Fake tensor conversion
  - Device/dtype detection from inputs
  - Memory efficient fusion
  - Graph flattening
  - Common compiler configurations

The utilities here are used by various backend implementations to handle
common operations and provide consistent behavior across different backends.
AOT autograd functionality is particularly important as it enables ahead-of-time
optimization of both forward and backward passes.
"""
log = ...
P = ParamSpec("P")
R = TypeVar("R")

class AotAutograd:
    def __init__(self, **kwargs: Any) -> None: ...
    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: Iterable[Any], **kwargs: Any
    ) -> Callable[..., Any]: ...

def aot_autograd(**kwargs: Any) -> AotAutograd: ...
def mem_efficient_fusion_kwargs(use_decomps: bool) -> dict[str, Any]: ...
def fake_tensor_unsupported(fn: Callable[[Any, list[Any], Any], R]) -> Any: ...
def device_from_inputs(example_inputs: Iterable[Any]) -> torch.device: ...
def dtype_from_inputs(example_inputs: Iterable[Any]) -> torch.dtype: ...
