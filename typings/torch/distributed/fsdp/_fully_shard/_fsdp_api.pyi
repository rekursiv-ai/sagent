from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

type _ReduceOp = dist.ReduceOp | dist.ReduceOp.RedOpType

@dataclass(frozen=True)
class MixedPrecisionPolicy:
    param_dtype: torch.dtype | None = ...
    reduce_dtype: torch.dtype | None = ...
    output_dtype: torch.dtype | None = ...
    cast_forward_inputs: bool = ...

class Comm(ABC):
    @abstractmethod
    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor: ...

class AllGather(Comm):
    @abstractmethod
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = ...,
    ) -> dist.Work | None: ...

class ReduceScatter(Comm):
    @abstractmethod
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = ...,
    ) -> dist.Work | None: ...

@dataclass
class OffloadPolicy: ...

@dataclass
class CPUOffloadPolicy(OffloadPolicy):
    pin_memory: bool = ...
