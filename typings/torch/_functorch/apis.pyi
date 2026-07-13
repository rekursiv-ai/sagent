from torch._functorch.utils import argnums_t, exposed_in
from torch._functorch.vmap import Callable, in_dims_t, out_dims_t

@exposed_in("torch.func")
def vmap(
    func: Callable,
    in_dims: in_dims_t = ...,
    out_dims: out_dims_t = ...,
    randomness: str = ...,
    *,
    chunk_size=...,
) -> Callable: ...
def chunk_vmap(
    func: Callable,
    in_dims: in_dims_t = ...,
    out_dims: out_dims_t = ...,
    randomness: str = ...,
    chunks=...,
) -> Callable: ...
@exposed_in("torch.func")
def grad(func: Callable, argnums: argnums_t = ..., has_aux: bool = ...) -> Callable: ...
@exposed_in("torch.func")
def grad_and_value(
    func: Callable, argnums: argnums_t = ..., has_aux: bool = ...
) -> Callable: ...
