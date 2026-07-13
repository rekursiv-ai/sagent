from collections.abc import Callable, Iterable
from typing import Any

from torch._inductor.autotune_process import (
    BenchmarkRequest,
    GPUDeviceBenchmarkMixin,
    TensorMeta,
)

import torch

log = ...

class ROCmBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        source_code: str,
    ) -> None: ...
    def precompile(self) -> None: ...
    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]: ...
    def update_workspace_size(self) -> None: ...
    def ensure_dll_loaded(self) -> None: ...
    def cleanup_run_fn(self) -> None: ...
