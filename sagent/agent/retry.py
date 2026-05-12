"""Retry classification, backoff, and send-with-retry loop.

Decides whether an error is transient (retry) or fatal (propagate),
extracts a server-advertised delay, and surfaces 429s as
``RateLimitError`` so interactive callers can pretty-print
"resumes at HH:MM:SS".

Constants:

- ``RETRY_BASE_SEC`` - exponential-backoff base (500 ms, doubled each attempt).
- ``MAX_RETRY_DELAY`` - normal-mode backoff cap.
- ``PERSISTENT_MAX_BACKOFF_SEC`` - persistent-mode cap (5 min) for
  unattended 429/529 loops.
- ``RETRYABLE_STATUS_CODES`` - explicit fast-path; anything ≥ 500 also
  retries via the numeric comparison in :func:`is_retryable`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import asyncio
import logging
import random
import time


if TYPE_CHECKING:
    import httpx
else:
    from sagent.lib.lazy_import import lazy_import

    httpx = lazy_import("httpx")  # 168ms

from sagent.custom_exceptions import StreamInterruptedError
from sagent.custom_types import (
    Model,
    ModelRequest,
    ModelResponse,
)


logger = logging.getLogger(__name__)

RETRY_BASE_SEC = 0.5
MAX_RETRY_DELAY = 32.0
PERSISTENT_MAX_BACKOFF_SEC = 300.0
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})

# Depth cap when unwinding ``__cause__`` chains - a pathological
# self-referencing cycle must not hang us.
_MAX_CAUSE_DEPTH = 5

_STREAM_FALLBACK_AFTER = 2
_MAX_STREAM_INTERRUPT_RETRIES = 2


class RetriesExhaustedError(Exception):
    """All retry attempts failed."""


class RateLimitError(Exception):
    """429 from the provider - raised immediately in interactive mode.

    Carries the reset timestamp (seconds since epoch) when the
    provider advertised one via ``retry-after`` or
    ``anthropic-ratelimit-unified-reset``. The message is
    pre-formatted for REPL display.

    Args:
      reset_time: Unix timestamp when the limit lifts, or ``None``.
      original: The underlying provider exception, retained for context.

    """

    def __init__(self, reset_time: float | None, original: Exception) -> None:
        self.reset_time = reset_time
        self.original = original
        if reset_time is not None and reset_time > time.time():
            clock = time.strftime("%H:%M:%S", time.localtime(reset_time))
            delta = reset_time - time.time()
            msg = f"Rate limited. Resumes at {clock} (~{delta:.0f}s)."
        else:
            msg = "Rate limited. Try again shortly."
        super().__init__(msg)


def is_retryable(error: Exception, model: Model) -> bool:
    """Classify an error as retryable (transient) or fatal.

    Asks the model's provider once (for SDK quirks the status-code
    path can't see); on miss, walks transport/status/cause across
    the cross-provider classifier. Providers receive the outer
    error and own walking their own ``__cause__`` chain.

    Args:
      error: Exception to classify.
      model: Active model; consulted for provider-specific edge cases.

    Returns:
      retryable: True if the error is transient.

    """
    if model.is_retryable_provider_error(error):
        return True
    return _is_retryable(error, 0)


def error_status(error: Exception) -> int | None:
    """Extract an HTTP status code from an error, recursively.

    Args:
      error: Exception (or chained cause) to inspect.

    Returns:
      status: HTTP status code, or None if not found.

    """
    return _error_status(error, 0)


def extract_retry_after(error: Exception) -> float | None:
    """Extract retry delay (sec) from an HTTP error response.

    Checks in order:

    1. ``retry-after`` (RFC 7231, seconds).
    2. ``anthropic-ratelimit-unified-reset`` (Unix timestamp when rate
       limit fully clears) - converted to delta-from-now.

    Args:
      error: Exception with an attached HTTP response.

    Returns:
      delay_sec: Seconds to wait, or None if no header found.

    """
    response = getattr(error, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", {}) or {}
    val = headers.get("retry-after")
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    reset = headers.get("anthropic-ratelimit-unified-reset")
    if reset is not None:
        try:
            return max(0.0, float(reset) - time.time())
        except (ValueError, TypeError):
            pass
    return None


async def send_with_retry(
    model: Model,
    request: ModelRequest,
    *,
    on_text: Callable[[str], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
    max_attempts: int,
    persistent_retry: bool,
    publish_recoverable: Callable[[str], None],
    on_discarded_response: Callable[[ModelResponse], None] | None = None,
) -> ModelResponse:
    """Send with backoff, error classification, stream fallback.

    On retryable errors: exponential backoff with jitter, respects
    Retry-After headers. After ``_STREAM_FALLBACK_AFTER`` consecutive
    streaming failures, falls back to non-streaming ``buffer()``.
    Tracks text already shown live on a failed first attempt and
    skips that prefix on retry.

    Args:
      model: Model to send the request to.
      request: Fully-built model request.
      on_text: Streaming callback for live text chunks.
      on_thinking: Streaming callback for live thinking chunks.
          Only fires on the first streaming attempt; ignored on
          retries (the buffered fallback has no separate thinking
          stream and the renderer would render thinking twice).
      max_attempts: Maximum number of retry attempts.
      persistent_retry: Enable persistent backoff for 429/529 errors.
      publish_recoverable: Callback for transient errors that recovered;
          each retry attempt invokes it with a ``multipart/x-error`` Message
          carrying the underlying exception and a structured stack trace.
      on_discarded_response: Called with the response from a completed
          request that will be retried (e.g. StreamInterruptedError).
          The API billed for these tokens; this callback lets the
          caller account them.

    Returns:
      response: Completed model response.

    Raises:
      RetriesExhaustedError: If all attempts fail.
      RateLimitError: On 429 in non-persistent mode.

    """
    stream_failures = 0
    last_error: Exception | None = None
    prior_emitted = ""
    attempt = -1
    persistent_attempt = 0
    stream_interrupts = 0
    while True:
        attempt += 1
        if attempt >= max_attempts:
            break
        use_stream = on_text is not None and stream_failures < _STREAM_FALLBACK_AFTER
        live = on_text if (use_stream and attempt == 0) else None
        # Thinking streams only on the first streaming attempt; on
        # retry we read thinking from the final response so the
        # renderer never sees the same content twice.
        live_thinking = on_thinking if (use_stream and attempt == 0) else None
        chunks: list[str] = []
        capture = _make_stream_callback(chunks, live)
        try:
            if use_stream:
                resp = await model.stream(
                    request=request,
                    on_text=capture,
                    on_thinking=live_thinking,
                )
                full = "".join(chunks)
            else:
                resp = await model.buffer(request=request)
                full = resp.message.text
            if on_text is not None and live is None and full:
                suffix = full.removeprefix(prior_emitted)
                if suffix:
                    on_text(suffix)
            return resp
        except StreamInterruptedError as e:
            stream_interrupts += 1
            prior_emitted = "".join(chunks) if live is not None else prior_emitted
            publish_recoverable(
                f"stream interrupted (attempt {attempt},"
                f" {stream_interrupts} interrupts so far): {e}"
            )
            if stream_interrupts > _MAX_STREAM_INTERRUPT_RETRIES:
                logger.warning(
                    "Stream indicated tool_use but delivered"
                    " no blocks after %d retries;"
                    " returning partial response.",
                    stream_interrupts - 1,
                )
                return e.response
            if on_discarded_response is not None:
                on_discarded_response(
                    e.response
                )  # may raise (e.g. budget exhaustion) — intentional
            attempt -= 1
            continue
        except Exception as e:
            if model.is_context_overflow(e):
                publish_recoverable(
                    f"context overflow (attempt {attempt}): {type(e).__name__}: {e}"
                )
                raise
            if not is_retryable(e, model):
                raise
            last_error = e
            if live is not None:
                prior_emitted = "".join(chunks)
            if use_stream:
                stream_failures += 1
            status = error_status(e)
            persistent = (
                persistent_retry
                and model.supports_persistent_retry
                and status in (429, 529)
            )
            server_delay = extract_retry_after(e)
            if status == 429 and not persistent:
                reset_time = (
                    time.time() + server_delay if server_delay is not None else None
                )
                raise RateLimitError(reset_time, e) from e
            if persistent:
                persistent_attempt += 1
                base = RETRY_BASE_SEC * (2.0**persistent_attempt)
                delay = min(
                    base + random.uniform(0, 0.25 * base),  # noqa: S311 -- jitter, not security
                    PERSISTENT_MAX_BACKOFF_SEC,
                )
                if server_delay is not None:
                    delay = max(delay, server_delay)
                attempt -= 1
            else:
                base = RETRY_BASE_SEC * (2.0**attempt)
                delay = min(
                    base + random.uniform(0, 0.25 * base),  # noqa: S311 -- jitter, not security
                    MAX_RETRY_DELAY,
                )
                if server_delay is not None:
                    delay = max(delay, server_delay)
            fallback = (
                " (falling back to non-streaming)"
                if (on_text is not None and stream_failures >= _STREAM_FALLBACK_AFTER)
                else ""
            )
            publish_recoverable(
                f"retry attempt {attempt}, waiting {delay:.1f}s"
                f"{fallback}: {type(e).__name__}: {e}"
            )
            logger.warning(
                "API error (attempt %d/%d): %s%s: %s. Retrying in %.0fs%s.",
                attempt + 1,
                max_attempts,
                type(e).__name__,
                f" {status}" if status is not None else "",
                e,
                delay,
                fallback,
            )
            if on_text is not None:
                on_text(
                    f"\n[error, retrying in {delay:.0f}s{fallback}...]\n",
                )
            await asyncio.sleep(delay)
    raise RetriesExhaustedError(
        f"Failed after {max_attempts} attempts: {last_error}",
    ) from last_error


def _is_retryable(error: Exception, depth: int) -> bool:
    """Walk transport/status/cause chain (depth-capped) for retryability."""
    if isinstance(error, (httpx.TransportError, ConnectionError, TimeoutError)):
        return True
    status = _error_status(error, 0)
    if status is not None and (status in RETRYABLE_STATUS_CODES or status >= 500):
        return True
    cause = error.__cause__
    if cause is not None and isinstance(cause, Exception) and depth < _MAX_CAUSE_DEPTH:
        return _is_retryable(cause, depth + 1)
    return False


def _error_status(error: Exception, depth: int) -> int | None:
    """Walk ``status_code`` / ``response.status_code`` / ``__cause__`` for a status."""
    status = getattr(error, "status_code", None)
    if status is None:
        resp = getattr(error, "response", None)
        status = getattr(resp, "status_code", None)
    if isinstance(status, int):
        return status
    cause = error.__cause__
    if cause is not None and isinstance(cause, Exception) and depth < _MAX_CAUSE_DEPTH:
        return _error_status(cause, depth + 1)
    return None


def _make_stream_callback(
    chunks: list[str],
    live: Callable[[str], None] | None,
) -> Callable[[str], None]:
    """Build a stream callback that captures chunks and forwards live ones."""
    if live is None:
        return chunks.append
    live_fn = live

    def _cb(c: str) -> None:
        chunks.append(c)
        live_fn(c)

    return _cb
