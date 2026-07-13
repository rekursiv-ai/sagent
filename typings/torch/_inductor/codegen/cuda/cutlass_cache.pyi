from typing import Any

import functools

from torch._inductor.utils import clear_on_fresh_cache

log = ...
CONFIG_PREFIX: str = ...

def get_config_request_key(
    arch: str, cuda_version: str, instantiation_level: str
) -> str: ...
@clear_on_fresh_cache
@functools.cache
def maybe_fetch_ops() -> list[Any] | None: ...
