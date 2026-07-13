from collections.abc import Callable, Sequence
from typing import Any
from typing_extensions import deprecated

import abc

import torch

HAS_NUMPY = ...

class ErrorMeta(Exception):
    def __init__(
        self, type: type[Exception], msg: str, *, id: tuple[Any, ...] = ...
    ) -> None: ...
    def to_error(self, msg: str | Callable[[str], str] | None = ...) -> Exception: ...

_DTYPE_PRECISIONS = ...

def default_tolerances(
    *inputs: torch.Tensor | torch.dtype,
    dtype_precisions: dict[torch.dtype, tuple[float, float]] | None = ...,
) -> tuple[float, float]: ...
def get_tolerances(
    *inputs: torch.Tensor | torch.dtype,
    rtol: float | None,
    atol: float | None,
    id: tuple[Any, ...] = ...,
) -> tuple[float, float]: ...
def make_scalar_mismatch_msg(
    actual: bool | complex,
    expected: bool | complex,
    *,
    rtol: float,
    atol: float,
    identifier: str | Callable[[str], str] | None = ...,
) -> str: ...
def make_tensor_mismatch_msg(
    actual: torch.Tensor,
    expected: torch.Tensor,
    matches: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    identifier: str | Callable[[str], str] | None = ...,
) -> str: ...

class UnsupportedInputs(Exception): ...

class Pair(abc.ABC):
    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple[Any, ...] = ...,
        **unknown_parameters: Any,
    ) -> None: ...
    @abc.abstractmethod
    def compare(self) -> None: ...
    def extra_repr(self) -> Sequence[str | tuple[str, Any]]: ...

class ObjectPair(Pair):
    def compare(self) -> None: ...

class NonePair(Pair):
    def __init__(self, actual: Any, expected: Any, **other_parameters: Any) -> None: ...
    def compare(self) -> None: ...

class BooleanPair(Pair):
    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple[Any, ...],
        **other_parameters: Any,
    ) -> None: ...
    def compare(self) -> None: ...

class NumberPair(Pair):
    _TYPE_TO_DTYPE = ...
    _NUMBER_TYPES = ...
    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple[Any, ...] = ...,
        rtol: float | None = ...,
        atol: float | None = ...,
        equal_nan: bool = ...,
        check_dtype: bool = ...,
        **other_parameters: Any,
    ) -> None: ...
    def compare(self) -> None: ...
    def extra_repr(self) -> Sequence[str]: ...

class TensorLikePair(Pair):
    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple[Any, ...] = ...,
        allow_subclasses: bool = ...,
        rtol: float | None = ...,
        atol: float | None = ...,
        equal_nan: bool = ...,
        check_device: bool = ...,
        check_dtype: bool = ...,
        check_layout: bool = ...,
        check_stride: bool = ...,
        **other_parameters: Any,
    ) -> None: ...
    def compare(self) -> None: ...
    def extra_repr(self) -> Sequence[str]: ...

def originate_pairs(
    actual: Any,
    expected: Any,
    *,
    pair_types: Sequence[type[Pair]],
    sequence_types: tuple[type, ...] = ...,
    mapping_types: tuple[type, ...] = ...,
    id: tuple[Any, ...] = ...,
    **options: Any,
) -> list[Pair]: ...
def not_close_error_metas(
    actual: Any,
    expected: Any,
    *,
    pair_types: Sequence[type[Pair]] = ...,
    sequence_types: tuple[type, ...] = ...,
    mapping_types: tuple[type, ...] = ...,
    **options: Any,
) -> list[ErrorMeta]: ...
def assert_close(
    actual: Any,
    expected: Any,
    *,
    allow_subclasses: bool = ...,
    rtol: float | None = ...,
    atol: float | None = ...,
    equal_nan: bool = ...,
    check_device: bool = ...,
    check_dtype: bool = ...,
    check_layout: bool = ...,
    check_stride: bool = ...,
    msg: str | Callable[[str], str] | None = ...,
) -> None: ...
@deprecated(
    "`torch.testing.assert_allclose()` is deprecated since 1.12 and will be removed in a future release. Please use `torch.testing.assert_close()` instead. You can find detailed upgrade instructions in https://github.com/pytorch/pytorch/issues/61844.",
    category=FutureWarning,
)
def assert_allclose(
    actual: Any,
    expected: Any,
    rtol: float | None = ...,
    atol: float | None = ...,
    equal_nan: bool = ...,
    msg: str = ...,
) -> None: ...
