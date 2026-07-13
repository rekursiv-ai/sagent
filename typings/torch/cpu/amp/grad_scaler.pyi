from typing_extensions import deprecated

import torch

__all__ = ["GradScaler"]

class GradScaler(torch.amp.GradScaler):
    @deprecated(
        "`torch.cpu.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cpu', args...)` instead.",
        category=FutureWarning,
    )
    def __init__(
        self,
        init_scale: float = ...,
        growth_factor: float = ...,
        backoff_factor: float = ...,
        growth_interval: int = ...,
        enabled: bool = ...,
    ) -> None: ...
