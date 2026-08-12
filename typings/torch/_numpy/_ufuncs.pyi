from collections.abc import Callable
from torch import Tensor
from typing import Any
from ._normalizations import (
    ArrayLike,
    ArrayLikeOrScalar,
    CastingModes,
    DTypeLike,
    NotImplementedType,
    OutArray,
    normalizer,
)

_binary = ...
NEP50_FUNCS = ...

def deco_binary_ufunc(torch_func) -> Callable[..., Tensor | Any]: ...
@normalizer
def matmul(
    x1: ArrayLike,
    x2: ArrayLike,
    /,
    out: OutArray | None = ...,
    *,
    casting: CastingModes | None = ...,
    order: NotImplementedType = ...,
    dtype: DTypeLike | None = ...,
    subok: NotImplementedType = ...,
    signature: NotImplementedType = ...,
    extobj: NotImplementedType = ...,
    axes: NotImplementedType = ...,
    axis: NotImplementedType = ...,
) -> Tensor: ...
@normalizer
def ldexp(
    x1: ArrayLikeOrScalar,
    x2: ArrayLikeOrScalar,
    /,
    out: OutArray | None = ...,
    *,
    where: NotImplementedType = ...,
    casting: CastingModes | None = ...,
    order: NotImplementedType = ...,
    dtype: DTypeLike | None = ...,
    subok: NotImplementedType = ...,
    signature: NotImplementedType = ...,
    extobj: NotImplementedType = ...,
) -> Tensor: ...
@normalizer
def divmod(
    x1: ArrayLike,
    x2: ArrayLike,
    out1: OutArray | None = ...,
    out2: OutArray | None = ...,
    /,
    out: tuple[OutArray | None, OutArray | None] = ...,
    *,
    where: NotImplementedType = ...,
    casting: CastingModes | None = ...,
    order: NotImplementedType = ...,
    dtype: DTypeLike | None = ...,
    subok: NotImplementedType = ...,
    signature: NotImplementedType = ...,
    extobj: NotImplementedType = ...,
) -> tuple[Tensor | Any, Tensor | Any]: ...
def modf(x, /, *args, **kwds) -> tuple[Tensor | Any, Tensor | Any]: ...

_binary = ...
_unary = ...
_fp_unary = ...

def deco_unary_ufunc(torch_func) -> Callable[..., Tensor | Any]: ...

__all__ = _binary + _unary
