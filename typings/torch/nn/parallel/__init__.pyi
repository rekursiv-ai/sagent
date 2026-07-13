from typing_extensions import deprecated as deprecated

from torch.nn.parallel.data_parallel import (
    DataParallel as DataParallel,
    data_parallel as data_parallel,
)
from torch.nn.parallel.distributed import (
    DistributedDataParallel as DistributedDataParallel,
)
from torch.nn.parallel.parallel_apply import parallel_apply as parallel_apply
from torch.nn.parallel.replicate import replicate as replicate
from torch.nn.parallel.scatter_gather import (
    gather as gather,
    scatter as scatter,
)

__all__ = [
    "DataParallel",
    "DistributedDataParallel",
    "data_parallel",
    "gather",
    "parallel_apply",
    "replicate",
    "scatter",
]

@deprecated(
    "`torch.nn.parallel.DistributedDataParallelCPU` is deprecated, please use `torch.nn.parallel.DistributedDataParallel` instead.",
    category=FutureWarning,
)
class DistributedDataParallelCPU(DistributedDataParallel): ...
