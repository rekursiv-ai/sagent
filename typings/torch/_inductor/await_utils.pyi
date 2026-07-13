from asyncio import AbstractEventLoop
from collections.abc import Awaitable, Coroutine, Generator, Iterator
from contextlib import contextmanager
from contextvars import Context
from typing import Any, Protocol, TypeVar

import asyncio

T = TypeVar("T")
type TCoro[T] = Generator[Any, None, T]

class TaskFactory(Protocol):
    def __call__(
        self,
        __loop: AbstractEventLoop,
        __factory: Coroutine[None, None, object] | Generator[None, None, object],
        __context: Context | None = ...,
        /,
    ) -> asyncio.futures.Future[object]: ...

TaskFactoryType = TaskFactory

def await_sync(awaitable: Awaitable[T]) -> T: ...
@contextmanager
def get_loop(always_create_new_loop: bool = ...) -> Iterator[AbstractEventLoop]: ...
