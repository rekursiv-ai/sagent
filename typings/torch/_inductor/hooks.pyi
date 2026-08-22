from collections.abc import Callable, Generator
from typing import Any

import contextlib

import torch

INTERMEDIATE_HOOKS: list[Callable[[str, torch.Tensor], None]] = ...

@contextlib.contextmanager
def intermediate_hook(fn) -> Generator[None, Any]: ...
def run_intermediate_hooks(name, val) -> None: ...
