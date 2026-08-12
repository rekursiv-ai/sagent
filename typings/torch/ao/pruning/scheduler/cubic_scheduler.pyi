from typing import Any, Literal
from .base_scheduler import BaseScheduler

__all__ = ["CubicSL"]

class CubicSL(BaseScheduler):
    def __init__(
        self,
        sparsifier,
        init_sl=...,
        init_t=...,
        delta_t=...,
        total_t=...,
        initially_zero=...,
        last_epoch=...,
        verbose=...,
    ) -> None: ...
    @staticmethod
    def sparsity_compute_fn(
        s_0, s_f, t, t_0, dt, n, initially_zero=...
    ) -> Literal[0]: ...
    def get_sl(self) -> list[Any | int]: ...
