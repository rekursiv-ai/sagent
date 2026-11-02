"""Retry a streamed reply that fails before its first event.

Typical usage example:

  async for event in retried(lambda: self._events(body)):
      yield event
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio

from sagent.model.exceptions import RetryableError


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


__all__ = [
    "retried",
]


async def retried[T](stream: Callable[[], AsyncIterator[T]]) -> AsyncIterator[T]:
    """Yield what ``stream`` yields, starting it again after a short early failure.

    Text the caller already showed cannot be taken back, and a long wait is the
    caller's to sit through or not: only a failure before the first event, with
    an advised wait of at most 30 seconds, is retried, at most four times.

    Args:
      stream: Starts one attempt.

    Yields:
      event: Each event of the attempt that streams.

    Raises:
      RetryableError: A failure after the first event, or before it with an
        advised wait longer than 30 seconds, or after five attempts.

    """
    attempt = 0
    while True:
        events = stream()
        try:
            first = await anext(events)
        except StopAsyncIteration:
            return
        except RetryableError as failure:
            if attempt == 4 or failure.retry_after_sec > 30.0:
                raise
            await asyncio.sleep(max(failure.retry_after_sec, 0.5 * (1 << attempt)))
            attempt += 1
            continue
        yield first
        async for event in events:
            yield event
        return
