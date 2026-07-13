from collections.abc import Callable, Sequence
from importlib.metadata import EntryPoint
from typing import Any, Protocol

from torch import fx

import torch

"""
This module implements TorchDynamo's backend registry system for managing compiler backends.

The registry provides a centralized way to register, discover and manage different compiler
backends that can be used with torch.compile(). It handles:

- Backend registration and discovery through decorators and entry points
- Lazy loading of backend implementations
- Lookup and validation of backend names
- Categorization of backends using tags (debug, experimental, etc.)

Key components:
- CompilerFn: Type for backend compiler functions that transform FX graphs
- _BACKENDS: Registry mapping backend names to entry points
- _COMPILER_FNS: Registry mapping backend names to loaded compiler functions

Example usage:
    @register_backend
    def my_compiler(fx_graph, example_inputs):
        # Transform FX graph into optimized implementation
        return compiled_fn

    # Use registered backend
    torch.compile(model, backend="my_compiler")

The registry also supports discovering backends through setuptools entry points
in the "torch_dynamo_backends" group. Example:
```
setup.py
---
from setuptools import setup

setup(
    name='my_torch_backend',
    version='0.1',
    packages=['my_torch_backend'],
    entry_points={
        'torch_dynamo_backends': [
            # name = path to entry point of backend implementation
            'my_compiler = my_torch_backend.compiler:my_compiler_function',
        ],
    },
)
```
```
my_torch_backend/compiler.py
---
def my_compiler_function(fx_graph, example_inputs):
    # Transform FX graph into optimized implementation
    return compiled_fn
```
Using `my_compiler` backend:
```
import torch

model = ...  # Your PyTorch model
optimized_model = torch.compile(model, backend="my_compiler")
```
"""
log = ...

class CompiledFn(Protocol):
    def __call__(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]: ...

type CompilerFn = Callable[[fx.GraphModule, list[torch.Tensor]], CompiledFn]
_BACKENDS: dict[str, EntryPoint | None] = ...
_COMPILER_FNS: dict[str, CompilerFn] = ...

def register_backend(
    compiler_fn: CompilerFn | None = ...,
    name: str | None = ...,
    tags: Sequence[str] = ...,
) -> Callable[..., Any]: ...

register_debug_backend = ...
register_experimental_backend = ...

def lookup_backend(compiler_fn: str | CompilerFn) -> CompilerFn: ...
def list_backends(exclude_tags=...) -> list[str]: ...
