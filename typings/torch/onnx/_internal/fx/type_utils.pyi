from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from typing_extensions import runtime_checkable

from torch._subclasses import fake_tensor

import onnx
import onnx.defs
import torch

"""Utilities for converting and operating on ONNX, JIT and torch types."""

@runtime_checkable
class TensorLike(Protocol):
    @property
    def dtype(self) -> torch.dtype | None: ...

def is_torch_complex_dtype(tensor_dtype: torch.dtype) -> bool: ...
def from_complex_to_float(dtype: torch.dtype) -> torch.dtype: ...
def from_sym_value_to_torch_dtype(sym_value: SYM_VALUE_TYPE) -> torch.dtype: ...
def is_optional_onnx_dtype_str(onnx_type_str: str) -> bool: ...
def from_torch_dtype_to_onnx_dtype_str(dtype: torch.dtype | type) -> set[str]: ...
def from_python_type_to_onnx_attribute_type(
    dtype: type, is_sequence: bool = ...
) -> onnx.defs.OpSchema.AttrType | None: ...
def is_torch_symbolic_type(value: Any) -> bool: ...
def from_torch_dtype_to_abbr(dtype: torch.dtype | None) -> str: ...
def from_scalar_type_to_torch_dtype(scalar_type: type) -> torch.dtype | None: ...

_TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS: dict[torch.dtype | type, set[str]] = ...
_OPTIONAL_ONNX_DTYPE_STR: set[str] = ...
_PYTHON_TYPE_TO_TORCH_DTYPE = ...
_COMPLEX_TO_FLOAT: dict[torch.dtype, torch.dtype] = ...
_SYM_TYPE_TO_TORCH_DTYPE = ...
_SCALAR_TYPE_TO_TORCH_DTYPE: dict[type, torch.dtype] = ...
_TORCH_DTYPE_TO_ABBREVIATION = ...
type SYM_VALUE_TYPE = torch.SymInt | torch.SymFloat | torch.SymBool
type META_VALUE_TYPE = fake_tensor.FakeTensor | SYM_VALUE_TYPE | int | float | bool
type BaseArgumentTypes = (
    str
    | int
    | float
    | bool
    | complex
    | torch.dtype
    | torch.Tensor
    | torch.device
    | torch.memory_format
    | torch.layout
    | torch._ops.OpOverload
    | torch.SymInt
    | torch.SymFloat
    | torch.SymBool
)
type Argument = (
    tuple[Argument, ...]
    | Sequence[Argument]
    | Mapping[str, Argument]
    | slice
    | range
    | torch.fx.Node
    | BaseArgumentTypes
    | None
)
