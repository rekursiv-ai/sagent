from dataclasses import dataclass

from torch import SymInt
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.types import PySymType
from torch.utils._backport_slots import dataclass_slots

import sympy
import torch

from .fake_tensor import _DispatchCacheKey, _MetadataIntLike

@dataclass_slots
@dataclass(frozen=True)
class _DeconstructedSymNode:
    _expr: sympy.Expr
    pytype: type
    _hint: int | float | bool | None
    constant: int | float | bool | None
    fx_node: torch.fx.Node
    @staticmethod
    def from_node(node: SymNode) -> _DeconstructedSymNode: ...
    def extract(self, shape_env: ShapeEnv) -> SymNode: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@dataclass_slots
@dataclass(frozen=True)
class _DeconstructedSymType:
    ty: type[PySymType]
    node: _DeconstructedSymNode
    @staticmethod
    def from_sym_type(value: PySymType) -> _DeconstructedSymType: ...
    def extract(self, shape_env: ShapeEnv) -> PySymType: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@dataclass_slots
@dataclass(frozen=True)
class _InputBackref:
    value: int

@dataclass_slots
@dataclass
class _PySymInputStub:
    value: PySymType | _DeconstructedSymType | _InputBackref
    def __init__(
        self, value: PySymType | _DeconstructedSymType | _InputBackref
    ) -> None: ...
    def strip_shape_env(self) -> None: ...
    def extract(self, shape_env: ShapeEnv) -> PySymType: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@dataclass_slots
@dataclass
class _SymIntOutputStub:
    value: int | _DeconstructedSymNode
    def __init__(self, value: SymInt, key_path: int | None) -> None: ...
    def extract(self, key: _DispatchCacheKey, shape_env: ShapeEnv) -> SymInt: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@dataclass_slots
@dataclass
class _CacheKeyState:
    sym_node_lookup: dict[int, int]
    known_symbols: set[sympy.Symbol]
    shape_env: ShapeEnv | None
    def __init__(self, shape_env: ShapeEnv | None = ...) -> None: ...
    def cache_on_shape_env(self) -> bool: ...
    def convert_sym_int(self, result: list[object], arg: SymInt) -> None: ...
    def convert_output(self, arg: _MetadataIntLike) -> _MetadataIntLike: ...
