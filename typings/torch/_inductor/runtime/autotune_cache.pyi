from typing import Any, override

import dataclasses

from torch.compiler._cache import CacheArtifact, CacheArtifactFactory

from .triton_compat import Config
from ..remote_cache import JsonDataTy, RemoteCache, RemoteCacheBackend

"""
PyTorch Inductor Autotuning Cache System

This module implements a caching system for autotuning configurations in PyTorch's Inductor compiler.
It provides mechanisms to store and retrieve optimal kernel configurations both locally and remotely,
which significantly speeds up compilation by reusing previously discovered optimal parameters.

The caching system includes:
- Local filesystem caching for individual machine reuse
- Remote caching for sharing optimizations across machines
- Bundled caching to efficiently store multiple related configurations
- Cache invalidation based on PyTorch versions and backend changes
- Serialization/deserialization support for worker processes

Key components:
- AutotuneCache: Main class for managing cache access and storage
- AutotuneCacheBundler: Bundles multiple cache entries for efficient storage
- LocalAutotuneCache: Handles filesystem-based caching
- _LocalAutotuneCacheBackend: Low-level file operations for cache storage
- AutotuneCacheArtifact: Integration with PyTorch's artifact system

This caching system is critical for performance as it eliminates the need to re-run
expensive autotuning operations when the same kernels are compiled multiple times.
"""
log = ...
type _InductorMetaTy = dict[str, object]

def inductor_meta_from_config() -> _InductorMetaTy: ...

@CacheArtifactFactory.register
class AutotuneCacheArtifact(CacheArtifact):
    @override
    def populate_cache(self) -> None: ...
    @override
    @staticmethod
    def type() -> str: ...
    @override
    @staticmethod
    def encode(content: JsonDataTy) -> bytes: ...

@dataclasses.dataclass
class AutotuneCache:
    configs_hash: str
    local_cache: tuple[RemoteCache[JsonDataTy], str] | None = ...
    remote_cache: tuple[RemoteCache[JsonDataTy], str] | None = ...
    @staticmethod
    def create(
        inductor_meta: _InductorMetaTy, filename: str, configs_hash: str
    ) -> AutotuneCache | None: ...
    def read_best(
        self, inductor_meta: _InductorMetaTy, configs: list[Config]
    ) -> Config | None: ...
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def save(
        self,
        config: Config,
        time_taken_ns: int,
        found_by_coordesc: bool = ...,
        triton_cache_hash: str | None = ...,
    ) -> None: ...

class _AutotuneCacheBundlerImpl:
    _key: str
    _cache: RemoteCache[JsonDataTy]
    _entries: dict[str, JsonDataTy]
    def end_compile(self) -> None: ...
    def put(self, basename: str, data: JsonDataTy) -> None: ...
    def __init__(self, key: str, cache: RemoteCache[JsonDataTy]) -> None: ...
    def sync(self) -> None: ...

class AutotuneCacheBundler:
    _bundler: _AutotuneCacheBundlerImpl | None = ...
    def __init__(self) -> None: ...
    @classmethod
    def begin_compile(
        cls,
        inductor_meta: _InductorMetaTy,
        *,
        code: str | None = ...,
        code_hash: str | None = ...,
    ) -> None: ...
    @classmethod
    def end_compile(cls) -> None: ...
    @classmethod
    def sync(cls) -> None: ...
    @classmethod
    def put(cls, filename: str, data: JsonDataTy) -> None: ...

class _LocalAutotuneCacheBackend(RemoteCacheBackend[bytes]): ...

class LocalAutotuneCache(RemoteCache[JsonDataTy]):
    def __init__(self) -> None: ...
