__all__ = ["hessian", "hvp", "jacobian", "jvp", "vhp", "vjp"]

def vjp(
    func, inputs, v=..., create_graph=..., strict=...
) -> tuple[
    Any | tuple[Any, ...] | tuple[Any] | tuple[tuple[Any, ...] | Any | tuple[Any], ...],
    Tensor
    | Any
    | tuple[Tensor, ...]
    | tuple[()]
    | tuple[Tensor]
    | tuple[Any, ...]
    | tuple[Tensor | Any, ...]
    | tuple[tuple[Tensor, ...] | tuple[()] | tuple[Tensor] | tuple[Any, ...], ...],
]: ...
def jvp(
    func, inputs, v=..., create_graph=..., strict=...
) -> tuple[
    Any | tuple[Any, ...] | tuple[Any] | tuple[tuple[Any, ...] | Any | tuple[Any], ...],
    Tensor
    | Any
    | tuple[Tensor, ...]
    | tuple[()]
    | tuple[Tensor]
    | tuple[Any, ...]
    | tuple[Tensor | Any, ...]
    | tuple[tuple[Tensor, ...] | tuple[()] | tuple[Tensor] | tuple[Any, ...], ...],
]: ...
def jacobian(
    func, inputs, create_graph=..., strict=..., vectorize=..., strategy=...
) -> (
    tuple[Any, ...]
    | list[Any]
    | Any
    | tuple[tuple[Any, ...], ...]
    | tuple[Any | tuple[Any, ...], ...]
    | tuple[tuple[Any, ...] | tuple[tuple[Any, ...], ...], ...]
    | Tensor
    | tuple[Tensor, ...]
): ...
def hessian(
    func,
    inputs,
    create_graph=...,
    strict=...,
    vectorize=...,
    outer_jacobian_strategy=...,
) -> (
    Any
    | tuple[Any, ...]
    | tuple[tuple[Any, ...], ...]
    | tuple[Any | tuple[Any, ...], ...]
    | list[Any]
    | tuple[tuple[Any, ...] | tuple[tuple[Any, ...], ...], ...]
): ...
def vhp(
    func, inputs, v=..., create_graph=..., strict=...
) -> tuple[
    Any | tuple[Any, ...] | tuple[Any] | tuple[tuple[Any, ...] | Any | tuple[Any], ...],
    Tensor
    | Any
    | tuple[Tensor, ...]
    | tuple[()]
    | tuple[Tensor]
    | tuple[Any, ...]
    | tuple[Tensor | Any, ...]
    | tuple[tuple[Tensor, ...] | tuple[()] | tuple[Tensor] | tuple[Any, ...], ...],
]: ...
def hvp(
    func, inputs, v=..., create_graph=..., strict=...
) -> tuple[
    Any | tuple[Any, ...] | tuple[Any] | tuple[tuple[Any, ...] | Any | tuple[Any], ...],
    Tensor
    | Any
    | tuple[Tensor, ...]
    | tuple[()]
    | tuple[Tensor]
    | tuple[Any, ...]
    | tuple[Tensor | Any, ...]
    | tuple[tuple[Tensor, ...] | tuple[()] | tuple[Tensor] | tuple[Any, ...], ...],
]: ...
