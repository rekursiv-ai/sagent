from abc import abstractmethod
from collections.abc import Callable
from typing import Any, NewType, Self, TypeVar, override

import dataclasses
import io
import pickle

from torch._guards import TracingContext
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, Tensor
from torch._subclasses.meta_utils import MetaTensorDesc, MetaTensorDescriber
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

import torch

_SymNodeT = TypeVar("_SymNodeT", torch.SymInt, torch.SymFloat)

@dataclasses.dataclass
class Options:
    ops_filter: Callable[[str], bool] | None = ...

class GraphPickler(pickle.Pickler):
    def __init__(self, file: io.BytesIO, options: Options | None = ...) -> None: ...
    @override
    def reducer_override(
        self, obj: object
    ) -> tuple[Callable[..., Any], tuple[Any, ...]]: ...
    @override
    def persistent_id(self, obj: object) -> str | None: ...
    @classmethod
    def dumps(cls, obj: object, options: Options | None = ...) -> bytes: ...
    @staticmethod
    def loads(data: bytes, fake_mode: FakeTensorMode) -> object: ...

class _UnpickleState:
    def __init__(self, fake_mode: FakeTensorMode) -> None: ...

_UnpickleStateToken = NewType("_UnpickleStateToken", object)

class _GraphUnpickler(pickle.Unpickler):
    def __init__(self, stream: io.BytesIO, unpickle_state: _UnpickleState) -> None: ...
    @override
    def persistent_load(self, pid: object) -> object: ...

class _ShapeEnvPickleData:
    data: dict[str, object]
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: ShapeEnv
    ) -> tuple[
        Callable[[Self, _UnpickleState], ShapeEnv], tuple[Self, _UnpickleStateToken]
    ]: ...
    def __init__(self, env: ShapeEnv) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> ShapeEnv: ...

class _SymNodePickleData:
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: _SymNodeT
    ) -> tuple[
        Callable[[Self, _UnpickleState], _SymNodeT], tuple[Self, _UnpickleStateToken]
    ]: ...
    def __init__(self, node: SymNode) -> None: ...
    def unpickle_sym_int(self, unpickle_state: _UnpickleState) -> torch.SymInt: ...

class _TensorPickleData:
    metadata: MetaTensorDesc[FakeTensor]
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: FakeTensor
    ) -> tuple[
        Callable[[Self, _UnpickleState], FakeTensor], tuple[Self, _UnpickleStateToken]
    ]: ...
    def __init__(self, describer: MetaTensorDescriber, t: Tensor) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> FakeTensor: ...

class _TorchNumpyPickleData:
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: object
    ) -> (
        tuple[
            Callable[[Self, _UnpickleState], object], tuple[Self, _UnpickleStateToken]
        ]
        | None
    ): ...
    def __init__(self, mod: str, name: str) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> Callable[..., object]: ...
    @classmethod
    def from_object(cls, tnp: object) -> Self | None: ...

class _GraphModulePickleData:
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: torch.fx.GraphModule
    ) -> tuple[
        Callable[[Self, _UnpickleState], torch.fx.GraphModule],
        tuple[Self, _UnpickleStateToken],
    ]: ...
    def __init__(self, gm: torch.fx.GraphModule, options: Options) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> torch.fx.GraphModule: ...

class _NodePickleData:
    def __init__(
        self,
        node: torch.fx.Node,
        mapping: dict[torch.fx.Node, _NodePickleData],
        options: Options,
    ) -> None: ...
    def unpickle(
        self,
        graph: torch.fx.Graph,
        mapping: dict[_NodePickleData, torch.fx.Node],
        unpickle_state: _UnpickleState,
    ) -> torch.fx.Node: ...

class _OpPickleData:
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, op: object
    ) -> tuple[Callable[[_UnpickleState], object], tuple[_UnpickleStateToken]]: ...
    @classmethod
    def pickle(cls, op: object, options: Options) -> _OpPickleData: ...
    @abstractmethod
    def unpickle(self, unpickle_state: _UnpickleState) -> object: ...

class _OpStrPickleData(_OpPickleData):
    def __init__(self, name: str) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> str: ...

class _OpOverloadPickleData(_OpPickleData):
    def __init__(self, name: str) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> torch._ops.OpOverload: ...

class _OpOverloadPacketPickleData(_OpPickleData):
    def __init__(self, name: str) -> None: ...
    def unpickle(
        self, unpickle_state: _UnpickleState
    ) -> torch._ops.OpOverloadPacket: ...

class _OpBuiltinPickleData(_OpPickleData):
    def __init__(self, root: str, name: str) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> object: ...

class _OpOperatorPickleData(_OpPickleData):
    def __init__(self, name: str) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> object: ...

class _GraphPickleData:
    def __init__(self, graph: torch.fx.Graph, options: Options) -> None: ...
    def unpickle(
        self, gm: torch.fx.GraphModule, unpickle_state: _UnpickleState
    ) -> torch.fx.Graph: ...

class _TracingContextPickleData:
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: torch._guards.TracingContext
    ) -> tuple[
        Callable[[Self, _UnpickleState], torch._guards.TracingContext],
        tuple[Self, _UnpickleStateToken],
    ]: ...
    def __init__(self, context: TracingContext) -> None: ...
    def unpickle(self, unpickle_state: _UnpickleState) -> TracingContext: ...
