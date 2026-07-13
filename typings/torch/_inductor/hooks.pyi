from collections.abc import Callable

import contextlib

import torch

INTERMEDIATE_HOOKS: list[Callable[[str, torch.Tensor], None]] = ...

@contextlib.contextmanager
def intermediate_hook(fn) -> Generator[None, Any, None]: ...
def run_intermediate_hooks(name, val) -> None: ...
