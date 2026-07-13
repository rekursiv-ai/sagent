from collections.abc import Callable, Sequence
from typing import TypeVar
from typing_extensions import ParamSpec

from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    NumberType,
    TensorLikeType,
)

_T = TypeVar("_T")
_P = ParamSpec("_P")

class elementwise_type_promotion_wrapper:
    def __init__(
        self,
        *,
        type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND,
        type_promoting_args: Sequence[str] | None = ...,
    ) -> None: ...
    def __call__(self, fn: Callable) -> Callable: ...

def is_cpu_scalar(x: TensorLikeType) -> bool: ...
def check_copy_devices(
    *, copy_from: TensorLikeType, copy_to: TensorLikeType
) -> None: ...
def out_wrapper(
    *out_names: str,
    exact_dtype: bool = ...,
    pass_is_out: bool = ...,
    preserve_memory_format: bool = ...,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
def backwards_not_supported(prim) -> _Wrapped[..., Any, ..., Any | None]: ...
def elementwise_unary_scalar_wrapper(
    fn: Callable[_P, _T],
) -> Callable[_P, _T | NumberType]: ...
