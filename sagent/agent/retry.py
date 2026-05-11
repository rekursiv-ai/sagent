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

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

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
from sagent.lib.message import response_text


logger = logging.getLogger(__name__)


RETRY_BASE_SEC = 0.5
MAX_RETRY_DELAY = 32.0
PERSISTENT_MAX_BACKOFF_SEC = 300.0
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
RETRYABLE_PROVIDER_ERROR_TYPES = frozenset({"rate_limit_error", "server_error"})

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


def is_retryable(error: Exception) -> bool:
    """Classify an error as retryable (transient) or fatal.

    Retry on: transport errors, 408/409/429, and anything ≥ 500.

    Args:
      error: Exception to classify.

    Returns:
      retryable: True if the error is transient.

    """
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
    log_event: Callable[..., None],
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
      log_event: Structured event logger.
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
                full = response_text(resp.content)
            if on_text is not None and live is None and full:
                suffix = full.removeprefix(prior_emitted)
                if suffix:
                    on_text(suffix)
            return resp
        except StreamInterruptedError as e:
            stream_interrupts += 1
            prior_emitted = "".join(chunks) if live is not None else prior_emitted
            log_event(
                "stream_interrupt_retry",
                attempt=attempt,
                interrupt_count=stream_interrupts,
                has_text=bool(
                    response_text(e.response.content).strip(),
                ),
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
                log_event("context_overflow", attempt=attempt)
                raise
            if not is_retryable(e):
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
            log_event(
                "retry",
                attempt=attempt,
                error=str(e),
                stream=use_stream,
                stream_failures=stream_failures,
                delay=round(delay, 1),
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
    if isinstance(error, (httpx.TransportError, ConnectionError, TimeoutError)):
        return True
    status = _error_status(error, 0)
    if status is not None and (status in RETRYABLE_STATUS_CODES or status >= 500):
        return True
    if _provider_marks_retryable(error):
        return True
    cause = error.__cause__
    if cause is not None and isinstance(cause, Exception) and depth < _MAX_CAUSE_DEPTH:
        return _is_retryable(cause, depth + 1)
    return False


def _provider_marks_retryable(error: Exception) -> bool:
    """Return True when a status-less provider error still declares retryability."""
    body = getattr(error, "body", None)
    if isinstance(body, Mapping):
        body_map = cast(Mapping[object, object], body)
        error_type = body_map.get("type")
        if isinstance(error_type, str) and error_type in RETRYABLE_PROVIDER_ERROR_TYPES:
            return True
    return "you can retry" in str(error).lower()


def _error_status(error: Exception, depth: int) -> int | None:
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
    if live is None:
        return chunks.append
    live_fn = live

    def _cb(c: str) -> None:
        chunks.append(c)
        live_fn(c)

    return _cb
