import functools

from torch._inductor.utils import clear_on_fresh_cache

log = ...

@clear_on_fresh_cache
@functools.lru_cache(1)
def get_cuda_arch() -> str | None: ...
@clear_on_fresh_cache
@functools.lru_cache(1)
def get_cuda_version() -> str | None: ...
@functools.cache
def nvcc_exist(nvcc_path: str | None = ...) -> bool: ...
