from collections.abc import Generator
from typing import Any

import contextlib

from torch import Tensor

def wrapper_set_seed(op, *args, **kwargs) -> Tensor: ...
@contextlib.contextmanager
def freeze_rng_state() -> Generator[None, Any]: ...
