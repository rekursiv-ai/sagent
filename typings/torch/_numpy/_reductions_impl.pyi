from typing import Any

from torch import Tensor

from ._normalizations import (
    ArrayLike,
    AxisLike,
    DTypeLike,
    KeepDims,
    NotImplementedType,
    OutArray,
)

"""Implementation of reduction operations, to be wrapped into arrays, dtypes etc
in the 'public' layer.

Anything here only deals with torch objects, e.g. "dtype" is a torch.dtype instance etc
"""

def count_nonzero(a: ArrayLike, axis: AxisLike = ..., *, keepdims: KeepDims = ...): ...
def argmax(
    a: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    *,
    keepdims: KeepDims = ...,
) -> Tensor: ...
def argmin(
    a: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    *,
    keepdims: KeepDims = ...,
) -> Tensor: ...
def any(
    a: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    keepdims: KeepDims = ...,
    *,
    where: NotImplementedType = ...,
): ...
def all(
    a: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    keepdims: KeepDims = ...,
    *,
    where: NotImplementedType = ...,
): ...
def amax(
    a: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    keepdims: KeepDims = ...,
    initial: NotImplementedType = ...,
    where: NotImplementedType = ...,
): ...

max = ...

def amin(
    a: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    keepdims: KeepDims = ...,
    initial: NotImplementedType = ...,
    where: NotImplementedType = ...,
): ...

min = ...

def ptp(
    a: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    keepdims: KeepDims = ...,
): ...
def sum(
    a: ArrayLike,
    axis: AxisLike = ...,
    dtype: DTypeLike | None = ...,
    out: OutArray | None = ...,
    keepdims: KeepDims = ...,
    initial: NotImplementedType = ...,
    where: NotImplementedType = ...,
): ...
def prod(
    a: ArrayLike,
    axis: AxisLike = ...,
    dtype: DTypeLike | None = ...,
    out: OutArray | None = ...,
    keepdims: KeepDims = ...,
    initial: NotImplementedType = ...,
    where: NotImplementedType = ...,
): ...

product = ...

def mean(
    a: ArrayLike,
    axis: AxisLike = ...,
    dtype: DTypeLike | None = ...,
    out: OutArray | None = ...,
    keepdims: KeepDims = ...,
    *,
    where: NotImplementedType = ...,
): ...
def std(
    a: ArrayLike,
    axis: AxisLike = ...,
    dtype: DTypeLike | None = ...,
    out: OutArray | None = ...,
    ddof=...,
    keepdims: KeepDims = ...,
    *,
    where: NotImplementedType = ...,
): ...
def var(
    a: ArrayLike,
    axis: AxisLike = ...,
    dtype: DTypeLike | None = ...,
    out: OutArray | None = ...,
    ddof=...,
    keepdims: KeepDims = ...,
    *,
    where: NotImplementedType = ...,
): ...
def cumsum(
    a: ArrayLike,
    axis: AxisLike = ...,
    dtype: DTypeLike | None = ...,
    out: OutArray | None = ...,
): ...
def cumprod(
    a: ArrayLike,
    axis: AxisLike = ...,
    dtype: DTypeLike | None = ...,
    out: OutArray | None = ...,
): ...

cumproduct = ...

def average(
    a: ArrayLike, axis=..., weights: ArrayLike = ..., returned=..., *, keepdims=...
) -> tuple[Any, Tensor | Any]: ...
def quantile(
    a: ArrayLike,
    q: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    overwrite_input=...,
    method=...,
    keepdims: KeepDims = ...,
    *,
    interpolation: NotImplementedType = ...,
): ...
def percentile(
    a: ArrayLike,
    q: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    overwrite_input=...,
    method=...,
    keepdims: KeepDims = ...,
    *,
    interpolation: NotImplementedType = ...,
): ...
def median(
    a: ArrayLike,
    axis=...,
    out: OutArray | None = ...,
    overwrite_input=...,
    keepdims: KeepDims = ...,
): ...
