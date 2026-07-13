from collections.abc import Callable, Generator
from typing import Any, NewType

import abc
import contextlib
import dataclasses
import functools
import types

from torch._dynamo.precompile_context import PrecompileCacheArtifact
from torch.compiler._cache import CacheArtifactFactory

"""
This module provides the infrastructure for creating and managing compile package
for torch.compile. We mainly have two abstractions here:
  - CompilePackage: Overarching data structure for store and lookup a list of compiled codes.
  - CodeCacheEntry: Data structure for a single code being compiled by torch.compile.
The caching behavior is always under user control explicitly so that a stronger guarantee can
be provided about cache hit for a specific compiled model. Users can load the compile package
from a different process or host.
"""
logger = ...

@dataclasses.dataclass(frozen=True)
class SerializedCode:
    co_argcount: int
    co_posonlyargcount: int
    co_kwonlyargcount: int
    co_nlocals: int
    co_stacksize: int
    co_flags: int
    co_code: bytes
    co_consts: tuple[Any, ...]
    co_names: tuple[str, ...]
    co_varnames: tuple[str, ...]
    co_filename: str
    co_name: str
    co_firstlineno: int
    co_cellvars: tuple[str, ...]
    co_freevars: tuple[str, ...]
    co_linetable: bytes | None = ...
    co_qualname: str | None = ...
    co_exceptiontable: bytes | None = ...
    co_lnotab: str | None = ...
    @classmethod
    @functools.cache
    def from_code_object(cls, code: types.CodeType) -> SerializedCode: ...
    @classmethod
    @functools.cache
    def to_code_object(cls, serialized_code: SerializedCode) -> types.CodeType: ...

@dataclasses.dataclass
class _GuardedCodeCacheEntry:
    guards_state: bytes
    dynamo_code: SerializedCode

_BackendId = NewType("_BackendId", str)
_FunctionId = NewType("_FunctionId", str)

@dataclasses.dataclass(frozen=True)
class InlinedSource:
    module: str
    firstlineno: int
    lastlineno: int
    checksum: str

@dataclasses.dataclass
class DynamoCaptureOutput:
    guarded_codes: list[_GuardedCodeCacheEntry]
    backend_ids: list[_BackendId]

@dataclasses.dataclass
class _DynamoCodeCacheEntry(DynamoCaptureOutput):
    python_code: SerializedCode
    python_module: str
    function_names: list[_FunctionId]
    import_sources: dict[str, str]
    code_source: str | None
    install_to_global: bool
    has_compile_id: bool = ...
    bypassed: bool = ...

@dataclasses.dataclass
class _DynamoCacheEntry:
    codes: list[_DynamoCodeCacheEntry]
    inlined_sources: set[InlinedSource]
    python_version: str = ...
    torch_version: str = ...
    @property
    def backend_ids(self) -> set[_BackendId]: ...

@CacheArtifactFactory.register
class _DynamoCacheArtifact(PrecompileCacheArtifact[_DynamoCacheEntry]):
    @staticmethod
    def type() -> str: ...
    def after_deserialization(self) -> _DynamoCacheEntry: ...

class CompilePackage:
    def __init__(
        self,
        fn: Callable[..., Any] | None,
        dynamo: _DynamoCacheEntry | None = ...,
        ignore_inlined_sources: bool = ...,
    ) -> None: ...
    def is_initialized(self) -> bool: ...
    def initialize(
        self,
        fn: Any,
        dynamo: _DynamoCacheEntry | None = ...,
        ignore_inlined_sources: bool = ...,
    ) -> None: ...
    @property
    def cached_backends(self) -> dict[_BackendId, Any]: ...
    @functools.cached_property
    def source_id(self) -> str: ...
    @contextlib.contextmanager
    def code_context(self, code: types.CodeType) -> Generator[None]: ...
    def add_guarded_code(
        self, guards_state: bytes, dynamo_code: types.CodeType
    ) -> None: ...
    def add_inlined_source(self, sources: list[types.CodeType]) -> None: ...
    def bypass_current_entry(self) -> None: ...
    def add_resume_function(
        self, python_code: types.CodeType, python_module: str, function_name: str | None
    ) -> None: ...
    def add_import_source(self, alias: str, module_name: str) -> None: ...
    def add_backend_id(self, backend_id: str, backend: Any | None = ...) -> None: ...
    def validate(self) -> None: ...
    def uninstall(self) -> None: ...
    def install(self, backends: dict[_BackendId, Any]) -> None: ...
    def cache_entry(self) -> _DynamoCacheEntry: ...
    @staticmethod
    def source_id_from_fn(fn: Callable[..., Any]) -> str: ...

@CacheArtifactFactory.register
class EagerCacheArtifact(PrecompileCacheArtifact[Any]):
    @staticmethod
    def type() -> str: ...
    def after_deserialization(self) -> Any: ...

type _Backends = dict[_BackendId, PrecompileCacheArtifact[Any]]

class DynamoStore(abc.ABC):
    def record_package(self, package: CompilePackage) -> None: ...
    def record_eager_backend(self, backend_id: _BackendId, backend: Any) -> None: ...
    @abc.abstractmethod
    def clear(self) -> None: ...
    @abc.abstractmethod
    def write(
        self, dynamo: _DynamoCacheEntry, backends: _Backends, path: str
    ) -> None: ...
    def save_cache_entry(self, cache_entry: _DynamoCacheEntry, key: str) -> None: ...
    def save_package(self, package: CompilePackage, key: str) -> None: ...
    @abc.abstractmethod
    def read(self, path: str) -> tuple[_DynamoCacheEntry, _Backends]: ...
    def load_cache_entry(
        self, key: str
    ) -> tuple[_DynamoCacheEntry, dict[_BackendId, Any]]: ...
    def load_package(
        self, fn: Any, key: str
    ) -> tuple[CompilePackage, dict[_BackendId, Any]]: ...

class InMemoryDynamoStore(DynamoStore):
    def __init__(self) -> None: ...
    def clear(self) -> None: ...
    def write(
        self, dynamo: _DynamoCacheEntry, backends: _Backends, path: str
    ) -> None: ...
    def read(self, path: str) -> tuple[_DynamoCacheEntry, _Backends]: ...

class DiskDynamoStore(DynamoStore):
    def __init__(self, path_prefix: str = ...) -> None: ...
    def clear(self) -> None: ...
    def write(
        self, dynamo: _DynamoCacheEntry, backends: _Backends, path: str
    ) -> None: ...
    def read(self, path: str) -> tuple[_DynamoCacheEntry, _Backends]: ...

class DiskDynamoCache(DiskDynamoStore):
    def save(self, package: CompilePackage) -> None: ...
    def load(
        self, fn: Callable[..., Any]
    ) -> tuple[_DynamoCacheEntry, dict[_BackendId, Any]] | None: ...
    def load_and_install_package(
        self, fn: Callable[..., Any]
    ) -> CompilePackage | None: ...

DynamoCache = ...
