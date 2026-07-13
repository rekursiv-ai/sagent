from collections.abc import Sequence

import torch.cuda

__all__ = ["all_gather", "all_reduce", "broadcast", "reduce", "reduce_scatter"]
SUM = ...

def is_available(tensors) -> bool: ...
def version() -> tuple[int, int, int] | tuple[int, int, int, str]: ...
def unique_id() -> bytes: ...
def init_rank(num_ranks, uid, rank) -> object: ...
def all_reduce(inputs, outputs=..., op=..., streams=..., comms=...) -> None: ...
def reduce(
    inputs: Sequence[torch.Tensor],
    output: torch.Tensor | Sequence[torch.Tensor] | None = ...,
    root: int = ...,
    op: int = ...,
    streams: Sequence[torch.cuda.Stream] | None = ...,
    comms=...,
    *,
    outputs: Sequence[torch.Tensor] | None = ...,
) -> None: ...
def broadcast(
    inputs: Sequence[torch.Tensor], root: int = ..., streams=..., comms=...
) -> None: ...
def all_gather(
    inputs: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
    streams=...,
    comms=...,
) -> None: ...
def reduce_scatter(
    inputs: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
    op: int = ...,
    streams=...,
    comms=...,
) -> None: ...
