from collections.abc import (
    Generator,
    Iterable as Iterable,
)
from enum import Enum
from typing import (
    Any,
    Union as Union,
)
from warnings import warn as warn

import contextlib

from torch.backends.cuda import (
    SDPAParams as SDPAParams,
    can_use_efficient_attention as can_use_efficient_attention,
    can_use_flash_attention as can_use_flash_attention,
)

"""This module contains functions and classes that alter the behavior of torch.nn.functional.scaled_dot_product_attention"""
__all__ = [
    "WARN_FOR_UNFUSED_KERNELS",
    "SDPAParams",
    "SDPBackend",
    "can_use_efficient_attention",
    "can_use_flash_attention",
    "sdpa_kernel",
]
WARN_FOR_UNFUSED_KERNELS = ...
_backend_names = ...

class SDPBackend(Enum):
    ERROR = ...
    MATH = ...
    FLASH_ATTENTION = ...
    EFFICIENT_ATTENTION = ...
    CUDNN_ATTENTION = ...
    OVERRIDEABLE = ...

@contextlib.contextmanager
def sdpa_kernel(
    backends: list[SDPBackend] | SDPBackend, set_priority: bool = ...
) -> Generator[dict[Any, Any], Any]: ...
