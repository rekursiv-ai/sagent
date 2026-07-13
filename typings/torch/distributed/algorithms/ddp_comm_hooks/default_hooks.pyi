from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

__all__ = [
    "allreduce_hook",
    "bf16_compress_hook",
    "bf16_compress_wrapper",
    "fp16_compress_hook",
    "fp16_compress_wrapper",
]

def allreduce_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]: ...
def fp16_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]: ...
def bf16_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]: ...
def fp16_compress_wrapper(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]],
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]: ...
def bf16_compress_wrapper(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]],
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]: ...
