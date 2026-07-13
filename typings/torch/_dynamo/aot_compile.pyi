from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import abc
import inspect
import types

import torch
import torch.fx

from .hooks import Hooks

log = ...

class SerializableCallable(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def serialize_compile_artifacts(cls, fn: Any) -> bytes: ...
    @classmethod
    @abc.abstractmethod
    def deserialize_compile_artifacts(cls, data: bytes) -> Any: ...

def bind_locals(
    signature: inspect.Signature, *args: Any, **kwargs: Any
) -> dict[str, Any]: ...

@dataclass
class CompileArtifacts:
    signature: inspect.Signature
    bytecode: types.CodeType
    guard_manager: torch._dynamo.guards.GuardManagerWrapper | None
    guards_state: bytes
    import_sources: dict[str, str]
    backend_id: str
    compiled_fn: SerializableCallable
    original_code: types.CodeType
    closure: tuple[Any, ...] | None

@dataclass
class AOTCompiledFunction:
    _artifacts: CompileArtifacts
    def guard_check(self, *args: Any, **kwargs: Any) -> bool: ...
    def __post_init__(self) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def save_compiled_function(self, path: str) -> None: ...
    @classmethod
    def serialize(cls, fn: AOTCompiledFunction) -> bytes: ...
    @classmethod
    def deserialize(cls, data: bytes) -> AOTCompiledFunction: ...

class BundledAOTAutogradSerializableCallable(SerializableCallable):
    def __init__(self, artifact: Any) -> None: ...
    def __getattr__(self, attr: Any) -> Any: ...
    @classmethod
    def from_backend_id(
        cls, backend_id: str
    ) -> BundledAOTAutogradSerializableCallable: ...
    @classmethod
    def serialize_compile_artifacts(
        cls, fn: BundledAOTAutogradSerializableCallable
    ) -> bytes: ...
    @classmethod
    def deserialize_compile_artifacts(cls, data: bytes) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

def aot_compile_fullgraph(
    model: Any,
    example_inputs: tuple[tuple[Any, ...], dict[str, Any]],
    hooks: Hooks,
    backend: Callable[[torch.fx.GraphModule, list[torch.Tensor]], SerializableCallable],
) -> AOTCompiledFunction: ...
