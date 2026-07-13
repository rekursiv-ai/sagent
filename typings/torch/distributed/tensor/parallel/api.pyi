from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle

__all__ = ["parallelize_module"]

def parallelize_module(
    module: nn.Module,
    device_mesh: DeviceMesh | None = ...,
    parallelize_plan: ParallelStyle | dict[str, ParallelStyle] | None = ...,
    *,
    src_data_rank: int | None = ...,
) -> nn.Module: ...
