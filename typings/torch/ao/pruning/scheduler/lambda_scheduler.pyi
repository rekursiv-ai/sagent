from collections.abc import Callable

from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from .base_scheduler import BaseScheduler

__all__ = ["LambdaSL"]

class LambdaSL(BaseScheduler):
    def __init__(
        self,
        sparsifier: BaseSparsifier,
        sl_lambda: Callable[[int], float] | list[Callable[[int], float]],
        last_epoch: int = ...,
        verbose: bool = ...,
    ) -> None: ...
    def get_sl(self) -> list[float]: ...
