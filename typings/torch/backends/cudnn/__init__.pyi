from collections.abc import Generator
from contextlib import contextmanager as contextmanager
from typing import (
    Any,
    Optional as Optional,
)

from torch._C import _cudnn as _cudnn
from torch.backends import (
    ContextProp as ContextProp,
    PropModule as PropModule,
    __allow_nonbracketed_mutation as __allow_nonbracketed_mutation,
    _FP32Precision as _FP32Precision,
    _get_fp32_precision_getter as _get_fp32_precision_getter,
    _set_fp32_precision_setter as _set_fp32_precision_setter,
)

__cudnn_version: int | None = ...
if _cudnn is not None: ...
else: ...

def version() -> int | None: ...

CUDNN_TENSOR_DTYPES = ...

def is_available() -> bool: ...
def is_acceptable(tensor) -> bool: ...
def set_flags(
    _enabled=...,
    _benchmark=...,
    _benchmark_limit=...,
    _deterministic=...,
    _allow_tf32=...,
    _fp32_precision=...,
) -> tuple[bool, bool, int | None, bool, bool, str]: ...
@contextmanager
def flags(
    enabled=...,
    benchmark=...,
    benchmark_limit=...,
    deterministic=...,
    allow_tf32=...,
    fp32_precision=...,
) -> Generator[None, Any]: ...

class CudnnModule(PropModule):
    def __init__(self, m, name) -> None: ...

    enabled = ...
    deterministic = ...
    benchmark = ...
    benchmark_limit = ...
    if is_available():
        benchmark_limit = ...
    allow_tf32 = ...
    conv = ...
    rnn = ...
    fp32_precision = ...

enabled: bool
deterministic: bool
benchmark: bool
allow_tf32: bool
benchmark_limit: int
