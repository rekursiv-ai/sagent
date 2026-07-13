from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather, ReduceScatter

import torch
import torch.distributed as dist

from ._fsdp_api import _ReduceOp
from ._fsdp_param import FSDPParam

class AllGatherResult(NamedTuple):
    all_gather_output: torch.Tensor
    all_gather_event: torch.Event | None
    all_gather_work: dist.distributed_c10d.Work | None
    param_all_gather_input_dtypes: list[list[torch.dtype]]
    param_all_gather_input_numels: list[list[int]]
    all_gather_input_split_sizes: list[int]

lib = ...

class DefaultAllocMixin:
    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor: ...

class ProcessGroupAllocMixin:
    def __init__(self, group: dist.ProcessGroup, *args: Any, **kwargs: Any) -> None: ...
    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor: ...

class DefaultAllGather(DefaultAllocMixin, AllGather):
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = ...,
    ) -> dist.Work | None: ...

class ProcessGroupAllocAllGather(ProcessGroupAllocMixin, AllGather):
    def __init__(self, group: dist.ProcessGroup) -> None: ...
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = ...,
    ) -> dist.Work | None: ...

class DefaultReduceScatter(DefaultAllocMixin, ReduceScatter):
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = ...,
    ) -> dist.Work: ...

class ProcessGroupAllocReduceScatter(ProcessGroupAllocMixin, ReduceScatter):
    def __init__(self, group: dist.ProcessGroup) -> None: ...
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = ...,
    ) -> dist.Work: ...

@torch.library.impl(lib, "all_gather_copy_in", "Meta")
def all_gather_copy_in_meta(
    all_gather_inputs: list[torch.Tensor],
    all_gather_output: torch.Tensor,
    inp_split_sizes: list[int],
    all_gather_input_numel: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]: ...
@torch.library.impl(lib, "all_gather_copy_in", "CUDA")
@torch.library.impl(lib, "all_gather_copy_in", "XPU")
@torch.library.impl(lib, "all_gather_copy_in", "HPU")
@torch.library.impl(lib, "all_gather_copy_in", "CPU")
@torch.library.impl(lib, "all_gather_copy_in", "MTIA")
@torch.library.impl(lib, "all_gather_copy_in", "PrivateUse1")
def all_gather_copy_in_cuda(
    all_gather_inputs: list[torch.Tensor],
    all_gather_output: torch.Tensor,
    inp_split_sizes: list[int],
    all_gather_input_numel: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]: ...
@torch.library.impl(lib, "split_with_sizes_copy", "Meta")
@torch.library.impl(lib, "split_with_sizes_copy", "CUDA")
@torch.library.impl(lib, "split_with_sizes_copy", "XPU")
@torch.library.impl(lib, "split_with_sizes_copy", "HPU")
@torch.library.impl(lib, "split_with_sizes_copy", "CPU")
@torch.library.impl(lib, "split_with_sizes_copy", "MTIA")
@torch.library.impl(lib, "split_with_sizes_copy", "PrivateUse1")
def split_with_sizes_copy(
    all_gather_output: torch.Tensor,
    all_gather_input_split_sizes: list[int],
    dim: int,
    out: list[torch.Tensor],
) -> None: ...
@torch.library.impl(lib, "chunk_cat", "Meta")
@torch.library.impl(lib, "chunk_cat", "CUDA")
@torch.library.impl(lib, "chunk_cat", "XPU")
@torch.library.impl(lib, "chunk_cat", "HPU")
@torch.library.impl(lib, "chunk_cat", "CPU")
@torch.library.impl(lib, "chunk_cat", "MTIA")
@torch.library.impl(lib, "chunk_cat", "PrivateUse1")
def chunk_cat(
    tensors: list[torch.Tensor], dim: int, num_chunks: int, out: torch.Tensor
) -> None: ...
@torch.no_grad()
def foreach_all_gather(
    fsdp_params: list[FSDPParam],
    group: dist.ProcessGroup,
    async_op: bool,
    all_gather_copy_in_stream: torch.Stream,
    all_gather_stream: torch.Stream,
    device: torch.device,
    all_gather_comm: AllGather,
) -> AllGatherResult | None: ...
@torch.no_grad()
def foreach_all_gather_copy_out(
    all_gather_result: AllGatherResult,
    fsdp_params: list[FSDPParam],
    group: dist.ProcessGroup,
) -> None: ...
@torch.no_grad()
def foreach_reduce(
    fsdp_params: list[FSDPParam],
    unsharded_grads: list[torch.Tensor],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    reduce_scatter_comm: ReduceScatter,
    orig_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    device: torch.device,
    gradient_divide_factor: float | None,
    all_reduce_group: dist.ProcessGroup | None,
    all_reduce_stream: torch.Stream,
    all_reduce_grads: bool,
    partial_reduce_output: torch.Tensor | None,
    all_reduce_hook: Callable[[torch.Tensor], None] | None,
    force_sum_reduction_for_comms: bool = ...,
) -> tuple[
    torch.Tensor,
    torch.Event,
    torch.Event,
    torch.Tensor | None,
    torch.Event | None,
    torch.Tensor | None,
]: ...
def foreach_reduce_scatter_copy_in(
    unsharded_grads: list[torch.Tensor],
    reduce_scatter_input: torch.Tensor,
    world_size: int,
) -> None: ...
