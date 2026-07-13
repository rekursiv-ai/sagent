from collections.abc import Iterator
from contextlib import AbstractContextManager

__all__ = ["loss_parallel"]

def loss_parallel() -> AbstractContextManager[Iterator[None]]: ...
