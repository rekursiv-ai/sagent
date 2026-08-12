from typing import TypeVar
from typing_extensions import ParamSpec

from torch import Tensor
from torch._prims_common import DimsType
from torch.masked.maskedtensor.core import MaskedTensor
from torch.types import _dtype as DType

type DimOrDims = DimsType | None
__all__: list[str] = ...
_T = TypeVar("_T")
_P = ParamSpec("_P")

def sum(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
def prod(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
def cumsum(
    input: Tensor, dim: int, *, dtype: DType | None = ..., mask: Tensor | None = ...
) -> Tensor: ...
def cumprod(
    input: Tensor, dim: int, *, dtype: DType | None = ..., mask: Tensor | None = ...
) -> Tensor: ...
def amax(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
def amin(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
def argmax(
    input: Tensor | MaskedTensor,
    dim: int | None = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
def argmin(
    input: Tensor | MaskedTensor,
    dim: int | None = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
def mean(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
def median(
    input: Tensor | MaskedTensor,
    dim: int = ...,
    *,
    keepdim: bool = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
def logsumexp(
    input: Tensor,
    dim: DimOrDims = ...,
    *,
    keepdim: bool = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
def logaddexp(
    input: Tensor | MaskedTensor,
    other: Tensor | MaskedTensor,
    *,
    dtype: DType | None = ...,
    input_mask: Tensor | None = ...,
    other_mask: Tensor | None = ...,
) -> Tensor: ...
def norm(
    input: Tensor | MaskedTensor,
    ord: float | None = ...,
    dim: DimOrDims = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
def var(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = ...,
    unbiased: bool | None = ...,
    *,
    correction: float | None = ...,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
def std(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = ...,
    unbiased: bool | None = ...,
    *,
    correction: int | None = ...,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
def softmax(
    input: Tensor | MaskedTensor,
    dim: int,
    *,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
def log_softmax(
    input: Tensor | MaskedTensor,
    dim: int,
    *,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
def softmin(
    input: Tensor | MaskedTensor,
    dim: int,
    *,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
def normalize(
    input: Tensor | MaskedTensor,
    ord: float,
    dim: int,
    *,
    eps: float = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
