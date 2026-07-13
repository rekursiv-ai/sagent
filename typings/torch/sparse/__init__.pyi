from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
    Any as Any,
    Optional as Optional,
    Union as Union,
)

from torch import Tensor as Tensor
from torch._C import (
    _add_docstr as _add_docstr,
    _sparse as _sparse,
)
from torch.types import _dtype as DType

from .semi_structured import (
    SparseSemiStructuredTensor as SparseSemiStructuredTensor,
    SparseSemiStructuredTensorCUSPARSELT as SparseSemiStructuredTensorCUSPARSELT,
    SparseSemiStructuredTensorCUTLASS as SparseSemiStructuredTensorCUTLASS,
    to_sparse_semi_structured as to_sparse_semi_structured,
)

type DimOrDims = int | tuple[int, ...] | list[int] | None
__all__ = [
    "SparseSemiStructuredTensor",
    "SparseSemiStructuredTensorCUSPARSELT",
    "SparseSemiStructuredTensorCUTLASS",
    "addmm",
    "as_sparse_gradcheck",
    "check_sparse_tensor_invariants",
    "log_softmax",
    "mm",
    "softmax",
    "solve",
    "sum",
    "to_sparse_semi_structured",
]
addmm = ...
mm = ...
sampled_addmm = ...

def sum(input: Tensor, dim: DimOrDims = ..., dtype: DType | None = ...) -> Tensor: ...

softmax = ...
spsolve = ...
log_softmax = ...
spdiags = ...

class check_sparse_tensor_invariants:
    @staticmethod
    def is_enabled() -> bool: ...
    @staticmethod
    def enable() -> None: ...
    @staticmethod
    def disable() -> None: ...
    def __init__(self, enable=...) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type, value, traceback) -> None: ...
    def __call__(self, mth) -> Callable[..., Any]: ...

def as_sparse_gradcheck(gradcheck) -> Callable[..., Any]: ...
