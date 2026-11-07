"""Retry a streamed reply that fails before its first event, paced per account.

Every call to one model of an account goes through its key in a shared ``Gate``,
so a rate limit one caller meets holds back every other caller on the host:
none sends into a 429 another has already met. A rate limit holds the key until
the server's advised time, or trips it into a doubling cooldown when the server
named none; a reply whose headers say a window is spent holds it until the
window resets.

Typical usage example:

  key = account_key("openai-sub", account="work", model="sol-6.1")
  async for event in retried(lambda: self._events(body), gate=gate, key=key):
      yield event
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import asyncio
import hashlib
import time

from wesearch.fetch.gate import GatePolicy

from sagent.model.exceptions import RateLimitedError, RetryableError


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable

    from wesearch.fetch.gate import Gate

    from sagent.providers.types import UsageWindow


__all__ = [
    "ACCOUNT_POLICY",
    "account_key",
    "observed",
    "retried",
]


ACCOUNT_POLICY: Final = GatePolicy(
    # A 429 that names no wait: 5s, doubling to 5 minutes.
    cooldown_sec=5.0,
    max_cooldown_sec=300.0,
    # A stream's first event can take minutes on a long context.
    probe_lease_sec=600.0,
    # Callers released together by one reset would meet the next 429 together.
    jitter_sec=2.0,
    # A longer hold is a spent window: the caller's to wait out, or move off.
    max_wait_sec=300.0,
)
"""How each account's key is paced and recovered."""


async def retried[T](
    stream: Callable[[], AsyncIterator[T]],
    *,
    gate: Gate,
    key: str,
    retries: int = 4,
    recheck_quota: bool = False,
) -> AsyncIterator[T]:
    """Yield what ``stream`` yields, starting it again after a short early failure.

    Text the caller already showed cannot be taken back, and a long wait is the
    caller's to sit through or not: only a failure before the first event, with
    an advised wait of at most 30 seconds, is retried, at most ``retries`` times.
    Each attempt first waits for ``gate`` to admit it to ``key``, and how it ends
    is recorded there, retried or not.

    Args:
      stream: Starts one attempt.
      gate: The pace and locks every caller on the host shares.
      key: The model and account the attempts are billed to, as ``account_key``.
      retries: The most attempts started again; ``0`` for a stream that cannot
        be, as one a process holds the conversation of.
      recheck_quota: Clear this key's cached hold once for a user-requested retry.

    Yields:
      event: Each event of the attempt that streams.

    Raises:
      RateLimitedError: The key is held longer than ``ACCOUNT_POLICY`` waits,
        so no request was sent; or a rate limit as ``RetryableError`` says.
      RetryableError: A failure after the first event, or before it with an
        advised wait longer than 30 seconds, or after five attempts.

    """
    if recheck_quota:
        await asyncio.to_thread(gate.reset, key, clear_hold=True)
    attempt = 0
    while True:
        await _admitted(gate, key)
        events = stream()
        try:
            first = await anext(events)
        except StopAsyncIteration:
            await asyncio.to_thread(gate.reset, key)
            return
        except RateLimitedError as failure:
            # The hold or cooldown is the backoff: the next admission sits it out.
            await _limited(gate, key, failure)
            if attempt == retries or failure.retry_after_sec > 30.0:
                raise
            attempt += 1
            continue
        except RetryableError as failure:
            await asyncio.to_thread(gate.release_probe, key)
            if attempt == retries or failure.retry_after_sec > 30.0:
                raise
            await asyncio.sleep(max(failure.retry_after_sec, 0.5 * (1 << attempt)))
            attempt += 1
            continue
        except BaseException:
            await asyncio.to_thread(gate.release_probe, key)
            raise
        await asyncio.to_thread(gate.reset, key)
        yield first
        async for event in events:
            yield event
        return


def account_key(
    section: str,
    *,
    model: str,
    account: str = "",
    credential: str = "",
) -> str:
    """Return the gate key of one model on the account a provider bills.

    A key is no wider than one model: an API key's limits are each model's, and
    a window named for a family (``weekly Opus``) leaves the account's other
    models free. A limit the whole account shares is then met once per model,
    and recorded for every model by the logins' ledger.

    Args:
      section: The provider's catalog section; providers of one section that
        bill one account share a key.
      model: The model called, as ``opus-5.5``.
      account: A named login's account; ``default`` for the default login.
      credential: What tells an API key's account apart, as the key itself; only a
        digest of it is kept, for the key names a file on disk.

    Returns:
      key: ``<section>@<account>/<model>``, the account a digest of
        ``credential`` when it has no name.

    """
    who = account or hashlib.sha256(credential.encode()).hexdigest()[:16]
    return f"{section}@{who}/{model}"


async def observed(gate: Gate, key: str, windows: Iterable[UsageWindow]) -> None:
    """Hold ``key`` until the last spent window a reply's headers report resets.

    Args:
      gate: The shared state.
      key: The model and account the reply was billed to, as ``account_key``.
      windows: The windows the reply reported.

    """
    until = max(
        (w.resets_at for w in windows if w.blocked and w.resets_at is not None),
        default=0.0,
    )
    if until > time.time():
        await asyncio.to_thread(gate.hold, key, until=until)


async def _admitted(gate: Gate, key: str) -> None:
    """Wait until ``gate`` admits a call to ``key``; refuse one held too long."""
    admission = await gate.admitted(key, ACCOUNT_POLICY)
    if not admission.admitted:
        raise RateLimitedError(
            f"The account is {admission.reason}.",
            retry_after_sec=admission.wait_sec,
        )


async def _limited(gate: Gate, key: str, failure: RateLimitedError) -> None:
    """Lock ``key`` for the wait the rate limit advised; a cooldown when it named none."""
    if failure.retry_after_sec > 0:
        until = time.time() + failure.retry_after_sec
        await asyncio.to_thread(gate.hold, key, until=until)
    else:
        await asyncio.to_thread(gate.trip, key, ACCOUNT_POLICY)
