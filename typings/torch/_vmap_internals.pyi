from collections.abc import Callable
from typing_extensions import deprecated

type in_dims_t = int | tuple
type out_dims_t = int | tuple[int, ...]

@deprecated(
    "Please use `torch.vmap` instead of `torch._vmap_internals.vmap`.",
    category=FutureWarning,
)
def vmap(
    func: Callable, in_dims: in_dims_t = ..., out_dims: out_dims_t = ...
) -> Callable: ...
