from collections.abc import Generator
from torch import Tensor
from typing import Any
import contextlib

def wrapper_set_seed(op, *args, **kwargs) -> Tensor: ...
@contextlib.contextmanager
def freeze_rng_state() -> Generator[None, Any, None]: ...
