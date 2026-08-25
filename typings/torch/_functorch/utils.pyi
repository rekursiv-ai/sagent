from collections.abc import Callable, Generator
from typing import Any

import contextlib

__all__ = [
    "argnums_t",
    "enable_single_level_autograd_function",
    "exposed_in",
    "unwrap_dead_wrappers",
]

@contextlib.contextmanager
def enable_single_level_autograd_function() -> Generator[None]: ...
def unwrap_dead_wrappers(args: tuple[Any, ...]) -> tuple[Any, ...]: ...

# Listed in ``__all__`` but never defined by the generator. An undefined
# decorator makes every function it decorates ``Unknown``, which erased
# ``torch.cond``'s signature at every call site.
def exposed_in[F](module: str) -> Callable[[F], F]: ...

type argnums_t = int | tuple[int, ...]
