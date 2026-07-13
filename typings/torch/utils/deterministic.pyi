import types

class _Deterministic(types.ModuleType):
    @property
    def fill_uninitialized_memory(self) -> bool: ...
    @fill_uninitialized_memory.setter
    def fill_uninitialized_memory(self, mode) -> None: ...
