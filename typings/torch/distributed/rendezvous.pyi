from collections.abc import Callable, Iterator

from torch.distributed import Store

_rendezvous_handlers: dict[str, Callable[..., Iterator[tuple[Store, int, int]]]] = ...
__all__ = ["register_rendezvous_handler", "rendezvous"]

def register_rendezvous_handler(scheme, handler) -> None: ...
def rendezvous(
    url: str, rank: int = ..., world_size: int = ..., **kwargs
) -> Iterator[tuple[Store, int, int]]: ...
