from collections.abc import Callable

from .base_sparsifier import BaseSparsifier

__all__ = ["WeightNormSparsifier"]

class WeightNormSparsifier(BaseSparsifier):
    def __init__(
        self,
        sparsity_level: float = ...,
        sparse_block_shape: tuple[int, int] = ...,
        zeros_per_block: int | None = ...,
        norm: Callable | int | None = ...,
    ) -> None: ...
    def update_mask(
        self,
        module,
        tensor_name,
        sparsity_level,
        sparse_block_shape,
        zeros_per_block,
        **kwargs,
    ) -> None: ...
