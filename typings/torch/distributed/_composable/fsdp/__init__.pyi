from collections.abc import Callable as Callable

from torch import nn as nn
from torch.distributed.device_mesh import DeviceMesh as DeviceMesh

import torch

class OffloadPolicy: ...

class CPUOffloadPolicy(OffloadPolicy):
    pin_memory: bool
    def __init__(self, pin_memory: bool = ...) -> None: ...

class MixedPrecisionPolicy:
    param_dtype: torch.dtype | None
    reduce_dtype: torch.dtype | None
    output_dtype: torch.dtype | None
    cast_forward_inputs: bool
    def __init__(
        self,
        param_dtype: torch.dtype | None = ...,
        reduce_dtype: torch.dtype | None = ...,
        output_dtype: torch.dtype | None = ...,
        cast_forward_inputs: bool = ...,
    ) -> None: ...

class FSDPModule: ...

class Shard:
    dim: int
    def __init__(self, dim: int) -> None: ...

def fully_shard(
    module: nn.Module,
    *,
    mesh: DeviceMesh | None = ...,
    reshard_after_forward: bool | int | None = ...,
    shard_placement_fn: Callable[[nn.Parameter], Shard | None] | None = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    offload_policy: OffloadPolicy = ...,
    ignored_params: set[nn.Parameter] | None = ...,
) -> FSDPModule: ...
def register_fsdp_forward_method(
    module: nn.Module,
    method_name: str,
) -> None: ...
