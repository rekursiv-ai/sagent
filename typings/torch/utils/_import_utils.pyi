from types import ModuleType

import functools

@functools.lru_cache
def dill_available() -> bool: ...
@functools.lru_cache
def import_dill() -> ModuleType | None: ...
