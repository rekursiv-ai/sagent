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

from sagent.lib.durations import humanize_duration
from sagent.types import runtime as runtime_types
from sagent.types.model import (
    Model,
    ModelRequest,
    ModelResponse,
    StreamInterruptedError,
)


logger = logging.getLogger(__name__)

RETRY_BASE_SEC = 0.5
MAX_RETRY_DELAY = 32.0
PERSISTENT_MAX_BACKOFF_SEC = 300.0
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})

# Depth cap when unwinding ``__cause__`` chains - a pathological
# self-referencing cycle must not hang us.
_MAX_CAUSE_DEPTH = 5

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
        self.diagnostics = error_diagnostics(original)
        if reset_time is not None and reset_time > time.time():
            clock = time.strftime("%H:%M:%S", time.localtime(reset_time))
            delta = reset_time - time.time()
            msg = f"Rate limited. Resumes at {clock} (~{humanize_duration(delta)})."
        else:
            msg = "Rate limited. Try again shortly."
        super().__init__(msg)
        if self.diagnostics:
            logger.info("RateLimitError diagnostics: %s", self.diagnostics)


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
            return _retry_after_seconds(float(val))
        except (ValueError, TypeError):
            pass
    reset = headers.get("anthropic-ratelimit-unified-reset")
    if reset is not None:
        try:
            return max(0.0, float(reset) - time.time())
        except (ValueError, TypeError):
            pass
    return None


# A ``Retry-After`` delta this large is implausible; values at or above it
# are an absolute Unix timestamp (epoch seconds), which some servers and
# proxies send instead of RFC 7231 delta-seconds. ~1 year in seconds; any
# real epoch (~1.8e9) clears it while no sane backoff approaches it.
_RETRY_AFTER_EPOCH_THRESHOLD_SEC = 365 * 24 * 60 * 60


def _retry_after_seconds(value: float) -> float:
    """Interpret a numeric ``Retry-After`` as a delay, converting epochs.

    Args:
      value: Parsed ``retry-after`` number (delta-seconds or epoch).

    Returns:
      delay_sec: Non-negative seconds to wait.

    """
    if value >= _RETRY_AFTER_EPOCH_THRESHOLD_SEC:
        return max(0.0, value - time.time())
    return max(0.0, value)


def error_diagnostics(error: Exception) -> str:
    """Render a one-line forensic summary of an HTTP error.

    Args:
      error: Exception to inspect, typically carrying ``.response``.

    Returns:
      summary: ``status=... headers={...} body=...`` or empty when
          there is nothing useful to surface.

    """
    snapshot = service_error_snapshot(error)
    parts: list[str] = []
    if snapshot.status is not None:
        parts.append(f"status={snapshot.status}")
    if snapshot.headers:
        parts.append(f"headers={dict(snapshot.headers)}")
    if snapshot.body:
        parts.append(f"body={snapshot.body!r}")
    return " ".join(parts)


def service_error_snapshot(error: Exception) -> runtime_types.ServiceErrorSnapshot:
    """Capture sanitized provider error details for durable runtime events.

    Args:
      error: Exception to inspect, typically carrying ``.response``.

    Returns:
      snapshot: JSON-serializable error details safe for session logs.

    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    return runtime_types.ServiceErrorSnapshot(
        type_name=type(error).__name__,
        message=str(error),
        status=error_status(error),
        headers=_diagnostic_headers(headers),
        body=_response_body_excerpt(response),
    )


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
    on_service_suspended: Callable[[float, float, bool, Exception], None] | None = None,
    resume_retry_at: float | None = None,
) -> ModelResponse:
    """Send with backoff and error classification.

    On retryable errors: exponential backoff with jitter, respects
    Retry-After headers. Tracks text already shown live on a failed
    first attempt and skips that prefix on retry.

    Args:
      model: Model to send the request to.
      request: Fully-built model request.
      on_text: Streaming callback for live text chunks. ``None``
          discards chunks; the request still streams under the hood
          (buffered transports are unreliable at large prompt sizes).
      on_thinking: Streaming callback for live thinking chunks.
          Only fires on the first attempt; on retry we read thinking
          from the final response so the renderer never sees the same
          content twice.
      max_attempts: Maximum number of retry attempts.
      persistent_retry: Enable persistent backoff for 429/529 errors.
      publish_recoverable: Callback for transient errors that recovered;
          each retry attempt invokes it with a ``multipart/x-error`` Message
          carrying the underlying exception and a structured stack trace.
      on_discarded_response: Called with the response from a completed
          request that will be retried (e.g. StreamInterruptedError).
          The API billed for these tokens; this callback lets the
          caller account them.
      on_service_suspended: Called when a recoverable provider error
          schedules a retry sleep. Arguments are retry_at, delay_sec,
          server_supplied, and the original exception.
      resume_retry_at: Optional wall-clock timestamp loaded from a prior
          service suspension; sleeps remaining time before the first send.

    Returns:
      response: Completed model response.

    Raises:
      RetriesExhaustedError: If all attempts fail.
      RateLimitError: On 429 in non-persistent mode.

    """
    if resume_retry_at is not None:
        delay = max(0.0, resume_retry_at - time.time())
        if delay > 0:
            await asyncio.sleep(delay)
    last_error: Exception | None = None
    prior_emitted = ""
    attempt = -1
    stream_attempt = 0
    persistent_attempt = 0
    stream_interrupts = 0
    while True:
        attempt += 1
        if attempt >= max_attempts:
            break
        live = on_text if attempt == 0 and not prior_emitted else None
        live_thinking = on_thinking if attempt == 0 and not prior_emitted else None
        chunks: list[str] = []
        capture = _make_stream_callback(chunks, live)
        stream_attempt += 1
        try:
            resp = await model.stream(
                request=request,
                on_text=capture,
                on_thinking=live_thinking,
            )
            full = "".join(chunks)
            if on_text is not None and live is None and full != prior_emitted:
                if full.startswith(prior_emitted):
                    suffix = full.removeprefix(prior_emitted)
                    if suffix:
                        on_text(suffix)
                else:
                    on_text("\n[retry response diverged; final response follows]\n")
                    on_text(full)
            return resp
        except StreamInterruptedError as e:
            stream_interrupts += 1
            prior_emitted = "".join(chunks) if live is not None else prior_emitted
            publish_recoverable(
                f"stream interrupted (attempt {stream_attempt},"
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
            status = error_status(e)
            # ``httpx.RemoteProtocolError`` ("peer closed connection without
            # sending complete message body") shows up when an upstream edge
            # cuts a streamed request body mid-flight -- typical on very
            # large compaction payloads. These carry no HTTP status, so the
            # normal 32s-cap loop bails after a handful of attempts. When
            # the caller opted into ``persistent_retry`` (compactor, batch
            # jobs), promote protocol errors into the long-backoff branch
            # so transient connection cuts don't fail the operation.
            is_protocol_flake = isinstance(e, httpx.RemoteProtocolError)
            persistent = (
                persistent_retry
                and model.supports_persistent_retry
                and (status in (429, 529) or is_protocol_flake)
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
            diagnostics = error_diagnostics(e)
            publish_recoverable(
                f"retry attempt {attempt}, waiting {delay:.1f}s:"
                f" {type(e).__name__}: {e}"
                + (f" [{diagnostics}]" if diagnostics else "")
            )
            logger.warning(
                "API error (attempt %d/%d): %s%s: %s. Retrying in %.0fs.%s",
                attempt + 1,
                max_attempts,
                type(e).__name__,
                f" {status}" if status is not None else "",
                e,
                delay,
                f" {diagnostics}" if diagnostics else "",
            )
            retry_at = time.time() + delay
            if on_service_suspended is not None:
                on_service_suspended(retry_at, delay, server_delay is not None, e)
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


_BODY_EXCERPT_CHARS = 500


def _diagnostic_headers(headers: object) -> dict[str, str]:
    """Return allowlisted response headers useful for rate-limit forensics."""
    if not isinstance(headers, Mapping):
        return {}
    out: dict[str, str] = {}
    for key, value in cast(Mapping[object, object], headers).items():
        name = str(key).lower()
        if (
            name == "retry-after"
            or name.startswith(("anthropic-ratelimit", "x-ratelimit"))
            or name in {"request-id", "x-request-id"}
        ):
            out[str(key)] = str(value)
    return out


def _response_body_excerpt(response: object) -> str:
    """Return up to ``_BODY_EXCERPT_CHARS`` of a response body, never raising.

    ``httpx.Response.text`` / ``.content`` are properties that raise
    ``httpx.ResponseNotRead`` when the body of a *streaming* response was
    never read -- which happens when an error carries a streaming response
    (e.g. Anthropic reports a ``rate_limit_error`` via an in-band SSE
    ``error`` event on a 200 stream). ``getattr`` does not suppress an
    exception raised inside the property, so the access itself is guarded.
    """
    if response is None:
        return ""
    try:
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return text[:_BODY_EXCERPT_CHARS]
        raw = getattr(response, "content", None)
    except httpx.ResponseNotRead:
        return ""
    if isinstance(raw, (bytes, bytearray)):
        try:
            return raw[:_BODY_EXCERPT_CHARS].decode("utf-8", errors="replace")
        except (AttributeError, ValueError):
            return ""
    return ""
