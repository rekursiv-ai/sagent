from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal

import dataclasses

from onnxscript import ir

import onnx

logger = ...

class _Empty:
    def __repr__(self) -> Literal[_EMPTY_DEFAULT]: ...

_EMPTY_DEFAULT = ...
_PY_TYPE_TO_ATTR_TYPE = ...
_LIST_TYPE_TO_ATTR_TYPE = ...
_ALL_VALUE_TYPES = ...
type TypeAnnotationValue = Any

@dataclasses.dataclass(frozen=True)
class TypeConstraintParam:
    name: str
    allowed_types: set[ir.TypeProtocol]
    description: str = ...
    def __hash__(self) -> int: ...
    @classmethod
    def any_tensor(cls, name: str, description: str = ...) -> TypeConstraintParam: ...
    @classmethod
    def any_value(cls, name: str, description: str = ...) -> TypeConstraintParam: ...

@dataclasses.dataclass(frozen=True)
class Parameter:
    name: str
    type_constraint: TypeConstraintParam
    required: bool
    variadic: bool
    default: Any = ...

    def has_default(self) -> bool: ...

@dataclasses.dataclass(frozen=True)
class AttributeParameter:
    name: str
    type: ir.AttributeType
    required: bool
    default: ir.Attr | None = ...

    def has_default(self) -> bool: ...

@dataclasses.dataclass
class OpSignature:
    domain: str
    name: str
    overload: str
    params: Sequence[Parameter | AttributeParameter]
    outputs: Sequence[Parameter]
    params_map: Mapping[str, Parameter | AttributeParameter] = ...
    opset_version: int | None = ...
    def __post_init__(self) -> None: ...
    def get(self, name: str) -> Parameter | AttributeParameter: ...
    def __contains__(self, name: str) -> bool: ...
    def __iter__(self) -> Iterator[Parameter | AttributeParameter]: ...
    @classmethod
    def from_opschema(cls, opschema: onnx.defs.OpSchema) -> OpSignature: ...
    @classmethod
    def from_function(
        cls,
        func,
        domain: str,
        name: str | None = ...,
        overload: str = ...,
        *,
        opset_version: int = ...,
    ) -> OpSignature: ...
