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

@_deco_axis_expand
def count_nonzero(a: ArrayLike, axis: AxisLike = ..., *, keepdims: KeepDims = ...): ...
@_deco_axis_expand
def argmax(
    a: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    *,
    keepdims: KeepDims = ...,
) -> Tensor: ...
@_deco_axis_expand
def argmin(
    a: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    *,
    keepdims: KeepDims = ...,
) -> Tensor: ...
@_deco_axis_expand
def any(
    a: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    keepdims: KeepDims = ...,
    *,
    where: NotImplementedType = ...,
): ...
@_deco_axis_expand
def all(
    a: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    keepdims: KeepDims = ...,
    *,
    where: NotImplementedType = ...,
): ...
@_deco_axis_expand
def amax(
    a: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    keepdims: KeepDims = ...,
    initial: NotImplementedType = ...,
    where: NotImplementedType = ...,
): ...

max = ...

@_deco_axis_expand
def amin(
    a: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    keepdims: KeepDims = ...,
    initial: NotImplementedType = ...,
    where: NotImplementedType = ...,
): ...

min = ...

@_deco_axis_expand
def ptp(
    a: ArrayLike,
    axis: AxisLike = ...,
    out: OutArray | None = ...,
    keepdims: KeepDims = ...,
): ...
@_deco_axis_expand
def sum(
    a: ArrayLike,
    axis: AxisLike = ...,
    dtype: DTypeLike | None = ...,
    out: OutArray | None = ...,
    keepdims: KeepDims = ...,
    initial: NotImplementedType = ...,
    where: NotImplementedType = ...,
): ...
@_deco_axis_expand
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

@_deco_axis_expand
def mean(
    a: ArrayLike,
    axis: AxisLike = ...,
    dtype: DTypeLike | None = ...,
    out: OutArray | None = ...,
    keepdims: KeepDims = ...,
    *,
    where: NotImplementedType = ...,
): ...
@_deco_axis_expand
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
@_deco_axis_expand
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
