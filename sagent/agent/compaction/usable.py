"""Usable: ask a model until its reply can be used, by one retry policy.

Every side call the agent makes for itself -- a summary, an audit, a fold --
asks the same way. An attempt that overflows shortens what is asked, when the
caller can, and otherwise ends the asking; a transient failure waits as the
vendor advised; a refusal, or a reply the caller cannot use, asks again, since
both come and go. Each counts one attempt, and the last attempt's failure is
raised as what it was, so a refusal is never reported as a context too long.

Typical usage example:

  summary = await usable(
      partial(model_call, request), read=summary_of, attempts=3, what="summary",
  )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import logging

from sagent.model.exceptions import (
    ContextOverflowError,
    RefusedError,
    RetryableError,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sagent.model.types import Response


__all__ = [
    "UnusableReplyError",
    "usable",
]


logger = logging.getLogger(__name__)


class UnusableReplyError(Exception):
    """The model answered, but not with anything the caller could use."""


async def usable[T](
    ask: Callable[[], Awaitable[Response]],
    *,
    read: Callable[[Response], T | None],
    attempts: int,
    what: str,
    shorten: Callable[[], None] | None = None,
) -> T:
    """Return what ``read`` makes of the first usable reply ``ask`` gets.

    Args:
      ask: Sends the request, as it stands now, and returns the reply.
      read: Returns what a reply is good for; ``None`` when it is not usable.
      attempts: The most calls of ``ask``.
      what: What is asked for, as the errors and the log name it.
      shorten: Makes what ``ask`` sends shorter after an overflow, raising when
        nothing shorter fits; ``None`` raises the overflow at once.

    Returns:
      value: What ``read`` returned.

    Raises:
      ContextOverflowError: An overflow ``shorten`` could not cure, or the last
        attempt overflowed.
      RetryableError: The last attempt failed for a while, as a rate limit does.
      RefusedError: The last attempt was refused; it says how many were.
      UnusableReplyError: The last attempt's reply was not usable.

    """
    failures: list[Exception] = []
    refused = 0
    for attempt in range(attempts):
        try:
            response = await ask()
        except ContextOverflowError as exc:
            if shorten is None:
                raise
            failures.append(exc)
            if attempt + 1 < attempts:
                shorten()
            continue
        except RetryableError as exc:
            failures.append(exc)
            logger.warning(
                "%s call failed (%s); waiting %ss",
                what,
                exc,
                exc.retry_after_sec,
            )
            if attempt + 1 < attempts:
                await asyncio.sleep(exc.retry_after_sec)
            continue
        if response.stop_reason == "refusal":
            refused += 1
            reason = response.stop_details or "no reason given"
            logger.warning("%s refused: %s", what, reason)
            failures.append(
                RefusedError(
                    f"The model refused {refused} of {attempts} attempts at the"
                    f" {what}: {reason}",
                ),
            )
            continue
        value = read(response)
        if value is not None:
            return value
        failures.append(
            UnusableReplyError(
                f"The last of {attempts} attempts returned no usable {what}.",
            ),
        )
    raise failures[-1]
