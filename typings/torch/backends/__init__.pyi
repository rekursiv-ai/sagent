from collections.abc import Generator
from typing import Any
from contextlib import contextmanager as contextmanager

import types

from torch.backends import (
    cpu as cpu,
    cuda as cuda,
    cudnn as cudnn,
    cusparselt as cusparselt,
    kleidiai as kleidiai,
    mha as mha,
    miopen as miopen,
    mkl as mkl,
    mkldnn as mkldnn,
    mps as mps,
    nnpack as nnpack,
    openmp as openmp,
    opt_einsum as opt_einsum,
    quantized as quantized,
)

__allow_nonbracketed_mutation_flag = ...

def disable_global_flags() -> None: ...
def flags_frozen() -> bool: ...

class ContextProp:
    def __init__(self, getter, setter) -> None: ...
    def __get__(self, obj, objtype): ...
    def __set__(self, obj, val) -> None: ...

class PropModule(types.ModuleType):
    def __init__(self, m, name) -> None: ...
    def __getattr__(self, attr): ...

class _FP32Precision:
    def __init__(self, backend, op) -> None: ...
    def __setattr__(self, name, value) -> None: ...
    def __getattr__(self, name) -> str: ...

def set_flags(_fp32_precision=...) -> tuple[str]: ...
@contextmanager
def flags(fp32_precision=...) -> Generator[None, Any, None]: ...

class GenericModule(PropModule):
    def __init__(self, m, name) -> None: ...

    fp32_precision = ...
