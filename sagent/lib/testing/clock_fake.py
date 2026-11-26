"""An asyncio event loop on a fake clock, for tests with short deadlines.

The clock moves only when nothing can run: it jumps to the next timer. A test whose
code waits out a 20 ms deadline on a real clock fails with a ``TimeoutError`` when a
loaded host stalls it longer; on this one the deadline fires after the code ran, every
run.

Every file the loop watches must be ready the moment its other end writes: this
process's own sockets and pipes, or a ``call_soon_threadsafe`` wake. A real child's
output arrives late, after the clock jumped past it.

Examples:
  asyncio.run(main(), loop_factory=FakeClockLoop)

"""

from __future__ import annotations

from typing import override

import asyncio
import selectors


__all__ = ["FakeClockLoop"]


class FakeClockLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock jumps to its next timer once nothing else can run."""

    def __init__(self) -> None:
        self._selector = _Selector()
        super().__init__(self._selector)

    @override
    def time(self) -> float:
        return self._selector.now


class _Selector(selectors.DefaultSelector):
    """A selector that skips a wait no file can end, adding it to the clock."""

    def __init__(self) -> None:
        super().__init__()
        self.now = 0.0

    @override
    def select(
        self,
        timeout: float | None = None,
    ) -> list[tuple[selectors.SelectorKey, int]]:
        ready = super().select(0)
        if ready or timeout == 0:
            return ready
        if timeout is None:
            return super().select(None)
        self.now += timeout
        return []
