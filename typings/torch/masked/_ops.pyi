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

@_apply_docstring_templates
def sum(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
@_apply_docstring_templates
def prod(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
@_apply_docstring_templates
def cumsum(
    input: Tensor, dim: int, *, dtype: DType | None = ..., mask: Tensor | None = ...
) -> Tensor: ...
@_apply_docstring_templates
def cumprod(
    input: Tensor, dim: int, *, dtype: DType | None = ..., mask: Tensor | None = ...
) -> Tensor: ...
@_apply_docstring_templates
def amax(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
@_apply_docstring_templates
def amin(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
@_apply_docstring_templates
def argmax(
    input: Tensor | MaskedTensor,
    dim: int | None = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
@_apply_docstring_templates
def argmin(
    input: Tensor | MaskedTensor,
    dim: int | None = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
@_apply_docstring_templates
def mean(
    input: Tensor | MaskedTensor,
    dim: DimOrDims = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
@_apply_docstring_templates
def median(
    input: Tensor | MaskedTensor,
    dim: int = ...,
    *,
    keepdim: bool = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
@_apply_docstring_templates
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
@_apply_docstring_templates
def norm(
    input: Tensor | MaskedTensor,
    ord: float | None = ...,
    dim: DimOrDims = ...,
    *,
    keepdim: bool | None = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
@_apply_docstring_templates
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
@_apply_docstring_templates
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
@_apply_docstring_templates
def softmax(
    input: Tensor | MaskedTensor,
    dim: int,
    *,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
@_apply_docstring_templates
def log_softmax(
    input: Tensor | MaskedTensor,
    dim: int,
    *,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
@_apply_docstring_templates
def softmin(
    input: Tensor | MaskedTensor,
    dim: int,
    *,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
@_apply_docstring_templates
def normalize(
    input: Tensor | MaskedTensor,
    ord: float,
    dim: int,
    *,
    eps: float = ...,
    dtype: DType | None = ...,
    mask: Tensor | None = ...,
) -> Tensor: ...
