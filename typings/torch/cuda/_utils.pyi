from typing import Any

import ctypes

class _CudaModule:
    def __init__(self, module: ctypes.c_void_p) -> None: ...
    def __getattr__(self, name: str) -> _CudaKernel: ...

class _CudaKernel:
    def __init__(self, func: ctypes.c_void_p, module: ctypes.c_void_p) -> None: ...
    def __call__(
        self,
        grid: tuple[int, int, int] = ...,
        block: tuple[int, int, int] = ...,
        args: list | None = ...,
        shared_mem: int = ...,
        stream: Any | None = ...,
    ) -> None: ...
